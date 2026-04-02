from smartController.neural_modules import DQN, PolicyNet, NEFENet, VariationalTransitionNet, NewTransitionNet
import torch.optim as optim
from collections import deque
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


class DAIF_Agent:
    def __init__(self, args):
        
        kwargs = args.intrusion_detection.to_dict()
        kwargs.update(args.neural_modules.to_dict())
        kwargs['use_packet_feats'] = args.use_packet_feats
        kwargs['node_features'] = args.node_features
        self.wbl = kwargs['wbl']
        self.action_size = int(kwargs['action_size'])
        self.neg_efe_net = NEFENet(kwargs)
        self.target_neg_efe_net = NEFENet(kwargs)
        self.update_target_model()
        self.efe_net_optimizer = optim.Adam(self.neg_efe_net.parameters(), lr=kwargs['learning_rate'])
        self.epistemic_regularisation_factor = float(kwargs['epistemic_regularisation_factor'])
        
        self.transitionnet = None
        self.transitionnet_optimizer = None
        self.variational_t_model = kwargs['variational_tmodel']

        state_size = int(kwargs['state_size'])
        hidden_state_size = int(kwargs['hidden_size']) + (int(kwargs['use_packet_feats']) * int(kwargs['hidden_size'])) + (int(kwargs['node_features']) * int(kwargs['hidden_size']))
        self.proprioceptive_state_size = state_size - hidden_state_size
        kwargs['proprioceptive_state_size'] = self.proprioceptive_state_size
        
        if kwargs['use_transition_model']:

            if self.variational_t_model:
                self.transitionnet = VariationalTransitionNet(kwargs)
                self.variational_variational_transition_loss = kwargs['variational_variational_transition_loss']
                self.kl_divergence_regularisation_factor = kwargs['transitionnet_kl_divergence_regularisation_factor']
            else:
                self.transitionnet = NewTransitionNet(kwargs)
                
            self.transitionnet_optimizer = optim.Adam(self.transitionnet.parameters(), lr=kwargs['learning_rate'])

        self.policynet = PolicyNet(kwargs)
        self.policynet_optimizer = optim.Adam(self.policynet.parameters(), lr=kwargs['learning_rate'])
        self.temperature_for_action_sampling = float(kwargs['temperature_for_action_sampling'])
        self.entropy_reg_coefficient = float(kwargs['entropy_reg_coefficient'])
        self.greedy_update = kwargs['greedy_update']
        self.memory_size = int(kwargs['agent_memory_size'])
        self.memory = [None] * self.memory_size
        self.memory_position = 0
        self.memory_size_actual = 0
        self.sequential_memory_size = int(kwargs['actor_train_interval_steps'])
        self.reset_sequential_memory()
        self.replay_batch_size = int(kwargs['replay_batch_size'])

        self.value_loss_fn = nn.MSELoss(reduction='mean')
        self.surrogate_policy_consistency = kwargs['surrogate_policy_consistency']
        self.use_critic_to_act = kwargs['use_critic_to_act']
        self._action_eye = torch.eye(self.action_size)


    def reset_sequential_memory(self):
        self.sequential_memory = deque(maxlen=self.sequential_memory_size)


    def update_target_model(self):
        self.target_neg_efe_net.load_state_dict(self.neg_efe_net.state_dict())


    def remember(self, state, action, reward, next_state, done, step):
        state_to_memorise = state.detach().clone()
        # print(id(state_to_memorise.untyped_storage()))
        next_state_to_memorise = next_state.detach().clone()
        # print(id(next_state_to_memorise.untyped_storage()))
        self.memory[self.memory_position] = (
            state_to_memorise, 
            action, 
            reward, 
            next_state_to_memorise,
            done)
        self.memory_position = (self.memory_position + 1) % self.memory_size
        self.memory_size_actual = min(self.memory_size_actual + 1, self.memory_size)

        self.sequential_memory.append(state_to_memorise)
        if len(self.sequential_memory) == self.sequential_memory_size:
            self.train_actor(step)

    
    def act(self, state):
        with torch.no_grad():
            if self.use_critic_to_act:
                neg_efe = self.neg_efe_net(state)
                action_probs = torch.softmax(
                    self.temperature_for_action_sampling * neg_efe,
                    dim=-1).squeeze()
            else:
                action_probs = self.policynet(state).squeeze()

            # sample from a categorical distribution
            action = torch.multinomial(action_probs, 1).item()

        return action
    

    def train_actor(self, step):
        self.reset_sequential_memory()

    def replay(self, step):
        """
        Use temporal difference on expected free energy to update the critic (efe bootstrapped network)
        This update is based on equation (17) of the Millidge's paper (after the sign correction), which in our paper is:
        \hat{G(s_t,a_t)} =  -r(o) -  \int Q(s)[logQ(s_t) - logQ(s_t|a_t, s_{t-1})] + G_\phi(s_t,a_t)
        which is equivalent to:
        -\hat{G(s_t,a_t)} =  r(o) +  \int Q(s)[logQ(s_t) - logQ(s_t|a_t, s_{t-1})] - G_\phi(s_t,a_t)
        This new form is more of a "value" (a policy's value is inversely prop. to the expected free energy)
        """
        if self.memory_size_actual < self.replay_batch_size:
            return

        self.neg_efe_net.eval()
        self.target_neg_efe_net.eval()
        self.policynet.train()
        self.transitionnet.train()
        
        indices = random.sample(range(self.memory_size_actual), self.replay_batch_size)
        minibatch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        action_onehots = torch.nn.functional.one_hot(actions, self.action_size).float()
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
        next_proprioceptive_states = next_states[:, -self.proprioceptive_state_size:]
        transition_inputs = torch.cat([states, action_onehots], dim=1)
            
        targets = rewards.clone()
        
        # 1. Forward passes for gain and consistency
        policy_probabilities = self.policynet(states)
        estimated_neg_efe_values = self.neg_efe_net(states)

        if self.variational_t_model:
            sample_next, eps_means, eps_logvars = self.transitionnet(transition_inputs)
            eps_var = eps_logvars.exp()
        else:
            predicted_observations = self.transitionnet(transition_inputs)

        # Vectorized computation of active epistemic gain
        with torch.no_grad():
            action_onehots_all = self._action_eye.to(states.device)
            expanded_actions = action_onehots_all.repeat(self.replay_batch_size, 1)  # [B*A, A]
            expanded_states = states.repeat_interleave(self.action_size, dim=0)  # [B*A, S]
            expanded_transition_inputs = torch.cat([expanded_states, expanded_actions], dim=1)

            if self.variational_t_model:
                _, l_eps_means, l_eps_logvars = self.transitionnet(expanded_transition_inputs)
                var = l_eps_logvars.exp()
                res = next_proprioceptive_states.repeat_interleave(self.action_size, 0) - l_eps_means
                log_likelihoods = -0.5 * ((res**2)/var + l_eps_logvars + 1.837877).sum(1).view(self.replay_batch_size, self.action_size) # 1.837877 is log(2*pi)
            else:
                predicted_nexts_all = self.transitionnet(expanded_transition_inputs) # [B*A, S']

                # Compute log P(o_next | o_current, a)
                log_likelihoods = -0.5 * F.mse_loss(
                    predicted_nexts_all,
                    next_proprioceptive_states.repeat_interleave(self.action_size, dim=0),
                    reduction='none'
                ).sum(dim=1).view(self.replay_batch_size, self.action_size) # [B, A]
            
            # Bayes' rule: Q(a|s,s') ∝ P(s'|s,a) * Q(a|s)
            log_posterior = log_likelihoods + torch.log(policy_probabilities.detach() + 1e-8)
            action_probs_posterior = torch.softmax(log_posterior, dim=1).clamp_min(1e-8)

            # KL divergence: Σ posterior * log(posterior/prior)
            active_epistemic_gains = (action_probs_posterior *
                    (torch.log(action_probs_posterior) - torch.log(policy_probabilities.detach() + 1e-8))
                    ).sum(dim=1, keepdim=True) # [B, 1]
            
            # active epistemic gain
            if self.variational_t_model:
                # These lines approximate the epistemic gain term: \int Q(s)[logQ(s_t) + logQ(s_t|a_t, s_{t-1})]
                # eps_means <- Q(s_t|a_t, s_{t-1})  {is a  reparameterisation in the variational setting} This is the "variational posterior's prior"
                # next_proprioceptive_states <- Q(s) {is interpreted as a sample from a spherical Gaussian centred on s in the variational setting}. This is the "variational posterior's posterior"
                # Analytical KL divergence.
                perceptive_epistemic_gains = 0.5 * torch.sum(
                    (1 / eps_var) + ((next_proprioceptive_states - eps_means) ** 2) / eps_var - 1 - eps_logvars,
                    dim=1, keepdim=True)
            else:
                perceptive_epistemic_gains = 0.5 * torch.sum((next_proprioceptive_states - predicted_observations) ** 2, dim=1, keepdim=True)

        targets += self.epistemic_regularisation_factor * (active_epistemic_gains + perceptive_epistemic_gains.detach())

        with torch.no_grad():
            estimated_next_neg_efe_values = self.target_neg_efe_net(next_states)
            if self.greedy_update:
                # DDQN style
                expected_next_neg_efe_values = estimated_next_neg_efe_values.max(1, keepdim=True)[0]
            else:
                if self.use_critic_to_act:
                    next_action_probs = torch.softmax(self.temperature_for_action_sampling * estimated_next_neg_efe_values, dim=1)
                else:
                    next_action_probs = self.policynet(next_states)
                expected_next_neg_efe_values = (next_action_probs * estimated_next_neg_efe_values).sum(dim=1, keepdim=True)

            targets += (~dones) * 0.99 * expected_next_neg_efe_values

            # Prepare targets for all actions
            target_neg_efes = self.neg_efe_net(states) # Re-compute for target base
            target_neg_efes[range(self.replay_batch_size), actions] = targets.squeeze()
                    
        # train the EFE value network (critic)
        self.neg_efe_net.train()
        value_loss = self.value_loss_fn(self.neg_efe_net(states), target_neg_efes)
        self.efe_net_optimizer.zero_grad()
        value_loss.backward()
        self.efe_net_optimizer.step()
    
        # perceptive and policy model training through VFE:
        self.neg_efe_net.eval()
        
        # The following corresponds Q(a_t | s_t) in eq. (6)
        if self.surrogate_policy_consistency:
            target_policy = torch.softmax(self.temperature_for_action_sampling * estimated_neg_efe_values.detach(), dim=1)
            policy_consistency = -0.5 * ((policy_probabilities - target_policy) ** 2).sum(dim=1).mean()
        else:
            # The following 2 loc's correspond p(a|s) according to eq. (8) in the same paper (Boltzman sampling)
            # i.e.: p(a|s) = \sigma(- \gamma G(s,a))
            efe_actions_log = torch.log_softmax(self.temperature_for_action_sampling * estimated_neg_efe_values.detach(), dim=1)
            # The following loc corresponds to the first term in eq (7), i.e.:
            # -E_{Q(s)}[ \int Q(a|s) logp(a|s) da]
            # This is the negative of the energy, i.e. the consitency of Q w.r.t p.
            # We need to maximise this energy by minimising VFE which is the negative of this fella.
            policy_consistency = torch.sum(policy_probabilities * efe_actions_log, dim=1).mean()
                    
        # The following 2 loc's correspond to the second term in eq (7), i.e.:
        # -E_{Q(s)}\{ H[Q(a|s)] \}
        # Also here, we want to maximise the entropy, that's why substract it from the loss.
        policy_log_probs = torch.log(torch.clamp(policy_probabilities, min=1e-8))
        policy_entropy = -(policy_probabilities * policy_log_probs).sum(1).mean()
        actor_loss = -policy_consistency - self.entropy_reg_coefficient * policy_entropy
        
        if self.variational_t_model:
            # Gaussian Log-likelihood.
            perceptive_consistency = -0.5 * torch.sum(((next_proprioceptive_states - eps_means) ** 2) / eps_var + eps_logvars + 1.837877, dim=1).mean() # 1.837877 is log(2*pi)
            # Batch variance
            batch_var = eps_var.mean(dim=0) + 1e-6
        else:
            # perceptive model
            # Perception consistency (cross entropy ≈ −MSE/2)
            perceptive_consistency = -0.5 * ((next_proprioceptive_states - predicted_observations) ** 2).sum(dim=1).mean()
            # Perception neutrality (entropy proxy using batch variance)
            batch_var = predicted_observations.var(dim=0) + 1e-6
        
        perceptive_entropy = 0.5 * torch.sum(torch.log(batch_var)) + 0.5 * next_proprioceptive_states.shape[1] * 2.837877 # 2.837877 is log(2*pi*e)
        
        perceptive_loss = -perceptive_entropy * self.entropy_reg_coefficient - perceptive_consistency

        vfe = actor_loss + perceptive_loss

        self.policynet_optimizer.zero_grad()
        self.transitionnet_optimizer.zero_grad()
        vfe.backward()
        self.transitionnet_optimizer.step()
        self.policynet_optimizer.step()
        

        if self.wbl: 
            self.wbl.log({
                'active_inference/value_loss': value_loss.item(),
                'active_inference/pragmatic_gain': rewards.mean().item(),
                'active_inference/epistemic_gain': perceptive_epistemic_gains.mean().item(),
                'active_inference/active_epistemic_gain': active_epistemic_gains.mean().item(),
                'active_inference/actor_loss': actor_loss.item(),
                'active_inference/perceptive_loss': perceptive_loss.item(),
                'active_inference/perceptive_entropy': perceptive_entropy.item(),
                'active_inference/perceptive_consistency': perceptive_consistency.item(),
                'active_inference/actor_entropy': policy_entropy.item(),
                'active_inference/actor_performance': policy_consistency.mean().item(),
                'active_inference/vfe': vfe.item()
            }, step=step)


class DAIP_Agent:
    def __init__(self, args):
        
        kwargs = args.intrusion_detection.to_dict()
        kwargs.update(args.neural_modules.to_dict())
        kwargs['use_packet_feats'] = args.use_packet_feats
        kwargs['node_features'] = args.node_features
        self.wbl = kwargs['wbl']
        self.action_size = int(kwargs['action_size'])
        self.neg_efe_net = NEFENet(kwargs)
        self.target_neg_efe_net = NEFENet(kwargs)
        self.update_target_model()
        self.efe_net_optimizer = optim.Adam(self.neg_efe_net.parameters(), lr=kwargs['learning_rate'])
        self.epistemic_regularisation_factor = float(kwargs['epistemic_regularisation_factor'])
        
        self.transitionnet = None
        self.transitionnet_optimizer = None
        self.variational_t_model = kwargs['variational_tmodel']

        state_size = int(kwargs['state_size'])
        hidden_state_size = int(kwargs['hidden_size']) + (int(kwargs['use_packet_feats']) * int(kwargs['hidden_size'])) + (int(kwargs['node_features']) * int(kwargs['hidden_size']))
        self.proprioceptive_state_size = state_size - hidden_state_size
        kwargs['proprioceptive_state_size'] = self.proprioceptive_state_size
        
        if kwargs['use_transition_model']:

            if self.variational_t_model:
                self.transitionnet = VariationalTransitionNet(kwargs)
                self.variational_variational_transition_loss = kwargs['variational_variational_transition_loss']
                self.kl_divergence_regularisation_factor = kwargs['transitionnet_kl_divergence_regularisation_factor']
            else:
                self.transitionnet = NewTransitionNet(kwargs)
                
            self.transitionnet_optimizer = optim.Adam(self.transitionnet.parameters(), lr=kwargs['learning_rate'])

        self.policynet = PolicyNet(kwargs)
        self.policynet_optimizer = optim.Adam(self.policynet.parameters(), lr=kwargs['learning_rate'])
        self.temperature_for_action_sampling = float(kwargs['temperature_for_action_sampling'])
        self.entropy_reg_coefficient = float(kwargs['entropy_reg_coefficient'])
        self.greedy_update = kwargs['greedy_update']
        self.memory_size = int(kwargs['agent_memory_size'])
        self.memory = [None] * self.memory_size
        self.memory_position = 0
        self.memory_size_actual = 0
        self.sequential_memory_size = int(kwargs['actor_train_interval_steps'])
        self.reset_sequential_memory()
        self.replay_batch_size = int(kwargs['replay_batch_size'])

        self.value_loss_fn = nn.MSELoss(reduction='mean')
        self.state_loss_fn = nn.MSELoss(reduction='mean')
        self.use_critic_to_act = kwargs['use_critic_to_act']

    def reset_sequential_memory(self):
        self.sequential_memory = deque(maxlen=self.sequential_memory_size)

    def update_target_model(self):
        self.target_neg_efe_net.load_state_dict(self.neg_efe_net.state_dict())


    def remember(self, state, action, reward, next_state, done, step):
        state_to_memorise = state.detach().clone()
        # print(id(state_to_memorise.untyped_storage()))
        next_state_to_memorise = next_state.detach().clone()
        # print(id(next_state_to_memorise.untyped_storage()))
        self.memory[self.memory_position] = (
            state_to_memorise, 
            action, 
            reward, 
            next_state_to_memorise,
            done)
        self.memory_position = (self.memory_position + 1) % self.memory_size
        self.memory_size_actual = min(self.memory_size_actual + 1, self.memory_size)

        self.sequential_memory.append(state_to_memorise)
        if len(self.sequential_memory) == self.sequential_memory_size:
            self.train_actor(step)

    
    def act(self, state):
        with torch.no_grad():
            if self.use_critic_to_act:
                neg_efe = self.neg_efe_net(state)
                action_probs = torch.softmax(
                    self.temperature_for_action_sampling * neg_efe,
                    dim=-1).squeeze()
            else:
                action_probs = self.policynet(state).squeeze()

            # sample from a categorical distribution
            action = torch.multinomial(action_probs, 1).item()

        return action
    

    def train_actor(self, step):
        """
        Trains the actor (policy network) by minimising the VFE.
        NOTE: THIS METHOD SOULD BE DONE ON-POLICY
        With respect to the paper's equation (6) (of Millidge's paper (DAI as Variational Policy Gradients).
        we correct the sign of the equation, i.e.:
        VFE = - \int Q(s)logp(o|s) + KL(Q(s)||p(s|s_{t-1},a_{t-1})] + E_{Q(s)}[KL[Q(a|s)||p(a|s)]

        - We do not have a POMDP but only an MDP, so we do not minimise accuracy over observations (\int Q(s)logp(o|s) = 1)
        - We are not touching the KL(Q(s)||p(s|s_{t-1},a_{t-1})] term either, because that will be the focus of the critic
        in the context of bootstrapping the EFE.
        - So we focus in  The last term in paper's eq (6) is E_{Q(s)}[KL[Q(a|s)||p(a|s)], which is itself divided into two terms:
            in eq. (7)
        """
        if self.use_critic_to_act:
            # If we are using the critic to act, we do not train the actor
            self.reset_sequential_memory()
            return
        
        self.neg_efe_net.eval()

        vfe = 0
        # batching the states
        states = torch.stack(list(self.sequential_memory))
        # The following corresponds Q(a_t | s_t) in eq. (6)
        policy_probabilities = self.policynet(states) 
        # The following 2 loc's correspond p(a|s) according to eq. (8) in the same paper (Boltzman sampling)
        # i.e.: p(a|s) = \sigma(- \gamma G(s,a))
        estimated_neg_efe_values = self.neg_efe_net(states).detach()
        efe_actions = torch.log_softmax(
            self.temperature_for_action_sampling * estimated_neg_efe_values, dim=1)

        # The following 2 loc's correspond to the first term in eq (7), i.e.:
        # -E_{Q(s)}[ \int Q(a|s) logp(a|s) da]
        # This is the negative of the energy, i.e. the consitency of Q w.r.t p.
        # We need to maximise this energy by minimising VFE which is the negative of this fella.
        energies = torch.sum(policy_probabilities * efe_actions, dim=1)
        vfe -= energies.mean()

        # The following 2 loc's correspond to the second term in eq (7), i.e.:
        # -E_{Q(s)}\{ H[Q(a|s)] \}
        # Also here, we want to maximise the entropy, that's why substract it from the loss.
        policy_log_probs = torch.log(torch.clamp(policy_probabilities, min=1e-8))
        policy_entropy = -(policy_probabilities * policy_log_probs).sum(1)
        expected_policy_entropy = policy_entropy.mean()
        vfe -= self.entropy_reg_coefficient * expected_policy_entropy

        self.policynet_optimizer.zero_grad()
        vfe.backward()
        self.policynet_optimizer.step()
        self.reset_sequential_memory()

        if self.wbl:
            self.wbl.log({
                'active_inference/actor_loss': vfe.item(),
                'active_inference/actor_entropy': expected_policy_entropy.item(),
                'active_inference/actor_performance': energies.mean().item()
            }, step=step)


    def replay(self, step):
        """
        Use temporal difference on expected free energy to update the critic (efe bootstrapped network)
        This update is based on equation (17) of the Millidge's paper (after the sign correction), which in our paper is:
        \hat{G(s_t,a_t)} =  -r(o) -  \int Q(s)[logQ(s_t) - logQ(s_t|a_t, s_{t-1})] + G_\phi(s_t,a_t)
        which is equivalent to:
        -\hat{G(s_t,a_t)} =  r(o) +  \int Q(s)[logQ(s_t) - logQ(s_t|a_t, s_{t-1})] - G_\phi(s_t,a_t)
        This new form is more of a "value" (a policy's value is inversely prop. to the expected free energy)
        """
        if self.memory_size_actual < self.replay_batch_size:
            return

        self.neg_efe_net.eval()
        self.target_neg_efe_net.eval()
        if self.transitionnet is not None: self.transitionnet.train()
        self.policynet.eval()

        indices = random.sample(range(self.memory_size_actual), self.replay_batch_size)
        minibatch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        action_onehots = torch.nn.functional.one_hot(actions, self.action_size).float()
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
        next_proprioceptive_states = next_states[:, -self.proprioceptive_state_size:]
            
        targets = rewards.clone()
        epistemic_gains = torch.zeros_like(rewards)
        
        if self.transitionnet is not None and self.epistemic_regularisation_factor > 0:
            transition_inputs = torch.cat([states, action_onehots], dim=1)
            
            # Vectorized computation of perceptive epistemic gain
            if self.variational_t_model:
                sample_next, eps_means, eps_logvars = self.transitionnet(transition_inputs)
                eps_var = eps_logvars.exp()
                # These lines approximate the epistemic gain term:
                # eps_means <- Q(s_t|a_t, s_{t-1})  {is a  reparameterisation in the variational setting} This is the "variational posterior's prior"
                # next_proprioceptive_states <- Q(s) {is interpreted as a sample from a spherical Gaussian centred on s in the variational setting}. This is the "variational posterior's posterior"
                # Analytical KL divergence.
                epistemic_gains = 0.5 * torch.sum((1 / eps_var) + ((next_proprioceptive_states - eps_means) ** 2) / eps_var - 1 - eps_logvars, dim=1, keepdim=True)
            else:
                predicted_observations = self.transitionnet(transition_inputs)
                epistemic_gains = 0.5 * torch.sum((next_proprioceptive_states - predicted_observations) ** 2, dim=1, keepdim=True)

            targets += self.epistemic_regularisation_factor * epistemic_gains.detach()

        with torch.no_grad():
            estimated_next_neg_efe_values = self.target_neg_efe_net(next_states)
            if self.greedy_update:
                # DDQN style
                expected_next_neg_efe_values = estimated_next_neg_efe_values.max(1, keepdim=True)[0]
            else:
                # Act-Inf style
                if self.use_critic_to_act:
                    # Value-based Act-Inf
                    next_action_probs = torch.softmax(self.temperature_for_action_sampling * estimated_next_neg_efe_values, dim=1)
                else:
                    # Act-Inf as policy gradients
                    next_action_probs = self.policynet(next_states)
                expected_next_neg_efe_values = (next_action_probs * estimated_next_neg_efe_values).sum(dim=1, keepdim=True)

            targets += (~dones) * 0.99 * expected_next_neg_efe_values

            # Prepare targets for all actions
            target_neg_efes = self.neg_efe_net(states)
            target_neg_efes[range(self.replay_batch_size), actions] = targets.squeeze()


        self.neg_efe_net.train()
        value_loss = self.value_loss_fn(self.neg_efe_net(states), target_neg_efes)
        self.efe_net_optimizer.zero_grad()
        value_loss.backward()
        self.efe_net_optimizer.step()
    
        if self.transitionnet is not None and self.epistemic_regularisation_factor > 0:
            if self.variational_t_model:
                if self.variational_variational_transition_loss:
                    reconstruction_loss = F.mse_loss(eps_means, next_proprioceptive_states, reduction='mean')
                    kl_div = 0.5 * torch.sum(eps_var + eps_means**2 - 1. - eps_logvars, dim=1).mean()
                    transition_loss = reconstruction_loss + self.kl_divergence_regularisation_factor * kl_div
                    if self.wbl:
                        self.wbl.log({
                            'active_inference/state_reconstruction_loss': reconstruction_loss.item(),
                            'active_inference/state kl_div': kl_div.item()
                        }, step=step)
                else:
                    transition_loss = self.state_loss_fn(sample_next, next_proprioceptive_states)
            else:
                transition_loss = self.state_loss_fn(predicted_observations, next_proprioceptive_states)

            self.transitionnet_optimizer.zero_grad()
            transition_loss.backward()
            self.transitionnet_optimizer.step()

            if self.wbl: 
                self.wbl.log({
                    'active_inference/transition_loss': transition_loss.item(),
                    'active_inference/epistemic_gain': epistemic_gains.mean().item()
                }, step=step)

        if self.wbl: 
            self.wbl.log({
                'active_inference/value_loss': value_loss.item(),
                'active_inference/pragmatic_gain': rewards.mean().item()
            }, step=step)
       

class DAIA_Agent:
    def __init__(self, args):
        
        kwargs = args.intrusion_detection.to_dict()
        kwargs.update(args.neural_modules.to_dict())
        kwargs['use_packet_feats'] = args.use_packet_feats
        kwargs['node_features'] = args.node_features
        self.wbl = kwargs['wbl']
        self.action_size = int(kwargs['action_size'])
        self.neg_efe_net = NEFENet(kwargs)
        self.target_neg_efe_net = NEFENet(kwargs)
        self.update_target_model()
        self.efe_net_optimizer = optim.Adam(self.neg_efe_net.parameters(), lr=kwargs['learning_rate'])
        self.epistemic_regularisation_factor = float(kwargs['epistemic_regularisation_factor'])
        
        self.transitionnet = None
        self.transitionnet_optimizer = None
        self.variational_t_model = kwargs['variational_tmodel']

        state_size = int(kwargs['state_size'])
        hidden_state_size = int(kwargs['hidden_size']) + (int(kwargs['use_packet_feats']) * int(kwargs['hidden_size'])) + (int(kwargs['node_features']) * int(kwargs['hidden_size']))
        self.proprioceptive_state_size = state_size - hidden_state_size
        kwargs['proprioceptive_state_size'] = self.proprioceptive_state_size

        if self.variational_t_model:
            self.transitionnet = VariationalTransitionNet(kwargs)
            self.variational_variational_transition_loss = kwargs['variational_variational_transition_loss']
            self.kl_divergence_regularisation_factor = kwargs['transitionnet_kl_divergence_regularisation_factor']
        else:
            self.transitionnet = NewTransitionNet(kwargs)
            
        self.transitionnet_optimizer = optim.Adam(self.transitionnet.parameters(), lr=kwargs['learning_rate'])

        self.policynet = PolicyNet(kwargs)
        self.policynet_optimizer = optim.Adam(self.policynet.parameters(), lr=kwargs['learning_rate'])
        self.temperature_for_action_sampling = float(kwargs['temperature_for_action_sampling'])
        self.entropy_reg_coefficient = float(kwargs['entropy_reg_coefficient'])
        self.greedy_update = kwargs['greedy_update']
        self.memory_size = int(kwargs['agent_memory_size'])
        self.memory = [None] * self.memory_size
        self.memory_position = 0
        self.memory_size_actual = 0
        self.sequential_memory_size = int(kwargs['actor_train_interval_steps'])
        self.reset_sequential_memory()
        self.replay_batch_size = int(kwargs['replay_batch_size'])

        self.value_loss_fn = nn.MSELoss(reduction='mean')
        self.use_critic_to_act = kwargs['use_critic_to_act']
        self._action_eye = torch.eye(self.action_size)

    def reset_sequential_memory(self):
        self.sequential_memory = deque(maxlen=self.sequential_memory_size)

    def update_target_model(self):
        self.target_neg_efe_net.load_state_dict(self.neg_efe_net.state_dict())


    def remember(self, state, action, reward, next_state, done, step):
        state_to_memorise = state.detach().clone()
        # print(id(state_to_memorise.untyped_storage()))
        next_state_to_memorise = next_state.detach().clone()
        # print(id(next_state_to_memorise.untyped_storage()))
        self.memory[self.memory_position] = (
            state_to_memorise, 
            action, 
            reward, 
            next_state_to_memorise,
            done)
        self.memory_position = (self.memory_position + 1) % self.memory_size
        self.memory_size_actual = min(self.memory_size_actual + 1, self.memory_size)

        self.sequential_memory.append(state_to_memorise)
        if len(self.sequential_memory) == self.sequential_memory_size:
            self.train_actor(step)

    
    def act(self, state):
        with torch.no_grad():
            if self.use_critic_to_act:
                neg_efe = self.neg_efe_net(state)
                action_probs = torch.softmax(
                    self.temperature_for_action_sampling * neg_efe,
                    dim=-1).squeeze()
            else:
                action_probs = self.policynet(state).squeeze()
            
            # sample from a categorical distribution
            action = torch.multinomial(action_probs, 1).item()

        return action
    

    def train_actor(self, step):
        # this trains not the actor but the perceptive model
        vfe = 0
        
        # batching the states
        states = torch.stack(list(self.sequential_memory))
        proprioceptive_states = states[:, -self.proprioceptive_state_size:]
        estimated_neg_efe_values = self.neg_efe_net(states).detach()
        efe_actions = torch.argmax(estimated_neg_efe_values, dim=1)
        action_onehots = torch.nn.functional.one_hot(efe_actions, self.action_size).float()
        transition_inputs = torch.cat([states, action_onehots], dim=1)

        self.transitionnet.train()
        
        if self.variational_t_model:
            _, eps_means, eps_logvars = self.transitionnet(transition_inputs)
            # Gaussian Log-likelihood.
            perceptive_consistency = -0.5 * torch.sum(
                ((proprioceptive_states[1:] - eps_means[:-1]) ** 2) / torch.exp(eps_logvars)
                + eps_logvars  + 1.837877, # 1.837877 is log(2*pi)
                dim=1,
            ).mean()
            # Batch variance
            batch_var = eps_logvars.exp().mean(dim=0) + 1e-6
        else:
            predicted_observations = self.transitionnet(transition_inputs)
            # Perception consistency (cross entropy ≈ −MSE/2)
            perceptive_consistency = -0.5 * ((proprioceptive_states[1:] - predicted_observations[:-1]) ** 2).sum(dim=1).mean()
            # Perception neutrality (entropy proxy using batch variance)
            batch_var = predicted_observations.var(dim=0) + 1e-6

        perceptive_entropy = 0.5 * torch.sum(torch.log(batch_var)) + 0.5 * proprioceptive_states.shape[1] * 2.837877 # 2.837877 is log(2*pi*e)

        vfe = - perceptive_entropy * self.entropy_reg_coefficient - perceptive_consistency
        if self.wbl:
            self.wbl.log({
                'active_inference/perceptive_entropy': perceptive_entropy.item(),
                'active_inference/perceptive_consistency': perceptive_consistency.item(),
                'active_inference/perceptive_loss': vfe.item()
            }, step=step)

        self.transitionnet_optimizer.zero_grad()
        vfe.backward()
        self.transitionnet_optimizer.step()
        self.reset_sequential_memory()


    def replay(self, step):
        """
        Use temporal difference on expected free energy to update the critic (efe bootstrapped network)
        This update is based on equation (17) of the Millidge's paper (after the sign correction), which in our paper is:
        \hat{G(s_t,a_t)} =  -r(o) -  \int Q(s)[logQ(s_t) - logQ(s_t|a_t, s_{t-1})] + G_\phi(s_t,a_t)
        which is equivalent to:
        -\hat{G(s_t,a_t)} =  r(o) +  \int Q(s)[logQ(s_t) - logQ(s_t|a_t, s_{t-1})] - G_\phi(s_t,a_t)
        This new form is more of a "value" (a policy's value is inversely prop. to the expected free energy)
        """
        if self.memory_size_actual < self.replay_batch_size:
            return

        self.neg_efe_net.eval()
        self.target_neg_efe_net.eval()
        self.transitionnet.eval()
        self.policynet.train()

        indices = random.sample(range(self.memory_size_actual), self.replay_batch_size)
        minibatch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
        next_proprioceptive_states = next_states[:, -self.proprioceptive_state_size:]
            
        targets = rewards.clone()
        
        # 1. Forward pass for policy prior
        action_probs_prior = self.policynet(states)

        # Vectorized computation of active epistemic gain
        with torch.no_grad():
            action_onehots_all = self._action_eye.to(states.device)
            expanded_actions = action_onehots_all.repeat(self.replay_batch_size, 1)  # [B*A, A]
            expanded_states = states.repeat_interleave(self.action_size, dim=0)  # [B*A, S]
            transition_inputs_all = torch.cat([expanded_states, expanded_actions], dim=1)

            if self.variational_t_model:
                _, l_eps_means, l_eps_logvars = self.transitionnet(transition_inputs_all)
                var = l_eps_logvars.exp()
                res = next_proprioceptive_states.repeat_interleave(self.action_size, 0) - l_eps_means
                log_likelihoods = -0.5 * ((res**2)/var + l_eps_logvars + 1.837877).sum(1).view(self.replay_batch_size, self.action_size) # 1.837877 is log(2*pi)
            else:
                predicted_nexts_all = self.transitionnet(transition_inputs_all)
                log_likelihoods = -0.5 * F.mse_loss(
                    predicted_nexts_all,
                    next_proprioceptive_states.repeat_interleave(self.action_size, dim=0),
                    reduction='none'
                ).sum(dim=1).view(self.replay_batch_size, self.action_size)
            
            # Bayes' rule: Q(a|s,s') ∝ P(s'|s,a) * Q(a|s)
            log_posterior = log_likelihoods + torch.log(action_probs_prior.detach() + 1e-8)
            action_probs_posterior = torch.softmax(log_posterior, dim=1).clamp_min(1e-8)

            # KL divergence: Σ posterior * log(posterior/prior)
            active_epistemic_gains = (action_probs_posterior *
                    (torch.log(action_probs_posterior) - torch.log(action_probs_prior.detach() + 1e-8))
                    ).sum(dim=1, keepdim=True) # [B, 1]

            # active epistemic gain

        targets += self.epistemic_regularisation_factor * active_epistemic_gains

        with torch.no_grad():
            estimated_next_neg_efe_values = self.target_neg_efe_net(next_states)
            if self.greedy_update:
                # DDQN style
                expected_next_neg_efe_values = estimated_next_neg_efe_values.max(1, keepdim=True)[0]
            else:
                if self.use_critic_to_act:
                    next_action_probs = torch.softmax(self.temperature_for_action_sampling * estimated_next_neg_efe_values, dim=1)
                else:
                    next_action_probs = self.policynet(next_states)
                expected_next_neg_efe_values = (next_action_probs * estimated_next_neg_efe_values).sum(dim=1, keepdim=True)

            targets += (~dones) * 0.99 * expected_next_neg_efe_values

            # Prepare targets for all actions
            target_neg_efes = self.neg_efe_net(states)
            target_neg_efes[range(self.replay_batch_size), actions] = targets.squeeze()

        self.neg_efe_net.train()
        value_loss = self.value_loss_fn(self.neg_efe_net(states), target_neg_efes)
        self.efe_net_optimizer.zero_grad()
        value_loss.backward()
        self.efe_net_optimizer.step()
    
        # train the policy network:
        target_logits = self.temperature_for_action_sampling * self.neg_efe_net(states).detach()
        target_policy = torch.softmax(target_logits, dim=1)
        policy_loss = -(target_policy * (action_probs_prior.clamp_min(1e-8).log())).sum(dim=1).mean()  # cross-entropy

        self.policynet_optimizer.zero_grad()
        policy_loss.backward()
        self.policynet_optimizer.step()
        
        if self.wbl: 
            self.wbl.log({
                'active_inference/policy_loss': policy_loss.item(),
                'active_inference/pragmatic_gain': rewards.mean().item(),
                'active_inference/value_loss': value_loss.item(),
                'active_inference/active_epistemic_gain': active_epistemic_gains.mean().item()
            }, step=step)

class DAISA_Agent:
    def __init__(self, args):
        
        kwargs = args.intrusion_detection.to_dict()
        kwargs.update(args.neural_modules.to_dict())
        kwargs['use_packet_feats'] = args.use_packet_feats
        kwargs['node_features'] = args.node_features
        self.wbl = kwargs['wbl']
        self.action_size = int(kwargs['action_size'])
        self.neg_efe_net = NEFENet(kwargs)
        self.target_neg_efe_net = NEFENet(kwargs)
        self.update_target_model()
        self.efe_net_optimizer = optim.Adam(self.neg_efe_net.parameters(), lr=kwargs['learning_rate'])
        self.epistemic_regularisation_factor = float(kwargs['epistemic_regularisation_factor'])

        state_size = int(kwargs['state_size'])
        hidden_state_size = int(kwargs['hidden_size']) + (int(kwargs['use_packet_feats']) * int(kwargs['hidden_size'])) + (int(kwargs['node_features']) * int(kwargs['hidden_size']))
        self.proprioceptive_state_size = state_size - hidden_state_size
        kwargs['proprioceptive_state_size'] = self.proprioceptive_state_size

        self.policynet = PolicyNet(kwargs)
        self.policynet_optimizer = optim.Adam(self.policynet.parameters(), lr=kwargs['learning_rate'])
        self.temperature_for_action_sampling = kwargs['temperature_for_action_sampling']
        self.entropy_reg_coefficient = kwargs['entropy_reg_coefficient']
        self.greedy_update = kwargs['greedy_update']
        self.memory_size = int(kwargs['agent_memory_size'])
        self.memory = [None] * self.memory_size
        self.memory_position = 0
        self.memory_size_actual = 0
        self.sequential_memory_size = kwargs['actor_train_interval_steps']
        self.reset_sequential_memory()
        self.replay_batch_size = int(kwargs['replay_batch_size'])
        self.value_loss_fn = nn.MSELoss(reduction='mean')
        self.use_critic_to_act = kwargs['use_critic_to_act']

    def reset_sequential_memory(self):
        self.sequential_memory = deque(maxlen=self.sequential_memory_size)

    def update_target_model(self):
        self.target_neg_efe_net.load_state_dict(self.neg_efe_net.state_dict())


    def remember(self, state, action, reward, next_state, done, step):
        state_to_memorise = state.detach().clone()
        # print(id(state_to_memorise.untyped_storage()))
        next_state_to_memorise = next_state.detach().clone()
        # print(id(next_state_to_memorise.untyped_storage()))
        self.memory[self.memory_position] = (
            state_to_memorise, 
            action, 
            reward, 
            next_state_to_memorise,
            done)
        self.memory_position = (self.memory_position + 1) % self.memory_size
        self.memory_size_actual = min(self.memory_size_actual + 1, self.memory_size)

        self.sequential_memory.append(state_to_memorise)
        if len(self.sequential_memory) == self.sequential_memory_size:
            self.train_actor(step)

    
    def act(self, state):
        with torch.no_grad():
            if self.use_critic_to_act:
                neg_efe = self.neg_efe_net(state)
                action_probs = torch.softmax(
                    self.temperature_for_action_sampling * neg_efe,
                    dim=-1).squeeze()
            else:
                action_probs = self.policynet(state).squeeze()

            # sample from a categorical distribution
            action = torch.multinomial(action_probs, 1).item()

        return action
    

    def train_actor(self, step):
        self.reset_sequential_memory()


    def replay(self, step):
        """
        Use temporal difference on expected free energy to update the critic (efe bootstrapped network)
        This update is based on equation (17) of the Millidge's paper (after the sign correction), which in our paper is:
        \hat{G(s_t,a_t)} =  -r(o) -  \int Q(s)[logQ(s_t) - logQ(s_t|a_t, s_{t-1})] + G_\phi(s_t,a_t)
        which is equivalent to:
        -\hat{G(s_t,a_t)} =  r(o) +  \int Q(s)[logQ(s_t) - logQ(s_t|a_t, s_{t-1})] - G_\phi(s_t,a_t)
        This new form is more of a "value" (a policy's value is inversely prop. to the expected free energy)
        """
        if self.memory_size_actual < self.replay_batch_size:
            return

        self.neg_efe_net.eval()
        self.target_neg_efe_net.eval()
        self.policynet.train()

        indices = random.sample(range(self.memory_size_actual), self.replay_batch_size)
        minibatch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

        targets = rewards.clone()

        # 1. Forward passes
        predicted_actions = self.policynet(states)
        estimated_neg_efe_values = self.neg_efe_net(states)
         
        with torch.no_grad():
            target_policy_inf = torch.softmax(self.temperature_for_action_sampling * estimated_neg_efe_values, dim=1)
            surrogate_active_epistemic_gains = torch.sum((predicted_actions.detach() - target_policy_inf) ** 2, dim=1, keepdim=True)

            targets += self.epistemic_regularisation_factor * surrogate_active_epistemic_gains

            estimated_next_neg_efe_values_tgt = self.target_neg_efe_net(next_states)
            if self.greedy_update:
                # DDQN style
                expected_next_neg_efe_values = estimated_next_neg_efe_values_tgt.max(1, keepdim=True)[0]
            else:
                if self.use_critic_to_act:
                    next_action_probs = torch.softmax(self.temperature_for_action_sampling * estimated_next_neg_efe_values_tgt, dim=1)
                else:
                    next_action_probs = self.policynet(next_states)
                expected_next_neg_efe_values = (next_action_probs * estimated_next_neg_efe_values_tgt).sum(dim=1, keepdim=True)

            targets += (~dones) * 0.99 * expected_next_neg_efe_values

            # Prepare targets for all actions
            target_neg_efes = self.neg_efe_net(states)
            target_neg_efes[range(self.replay_batch_size), actions] = targets.squeeze()


        self.neg_efe_net.train()
        value_loss = self.value_loss_fn(self.neg_efe_net(states), target_neg_efes)
        self.efe_net_optimizer.zero_grad()
        value_loss.backward()
        self.efe_net_optimizer.step()
    

        # train the policy network:
        target_policy_vfe = torch.softmax(self.temperature_for_action_sampling * estimated_neg_efe_values.detach(), dim=1)
        policy_loss = -(target_policy_vfe * (predicted_actions.clamp_min(1e-8).log())).sum(dim=1).mean()  # cross-entropy
        self.policynet_optimizer.zero_grad()
        policy_loss.backward()
        self.policynet_optimizer.step()

        if self.wbl: 
            self.wbl.log({
                'active_inference/policy_loss': policy_loss.item(),
                'active_inference/pragmatic_gain': rewards.mean().item(),
                'active_inference/value_loss': value_loss.item(),
                'active_inference/active_epistemic_gain': surrogate_active_epistemic_gains.mean().item()
            }, step=step)


class ValueLearningAgent:
    
    def __init__(self, args):

        kwargs = args.intrusion_detection.to_dict()
        kwargs.update(args.neural_modules.to_dict())
        kwargs['use_packet_feats'] = args.use_packet_feats
        
        self.wbl = kwargs['wbl']
        self.state_size = int(kwargs['state_size'])
        self.action_size = int(kwargs['action_size'])
        self.memory_size = int(kwargs['agent_memory_size'])
        self.memory = [None] * self.memory_size
        self.memory_position = 0
        self.memory_size_actual = 0
        self.gamma = float(kwargs['agent_discount_rate'])  # discount rate
        self.boltzmann_sampling = kwargs['boltzmann_sampling']
        self.epsilon = float(kwargs['init_epsilon_egreedy'])  # exploration rate
        self.epsilon_min = float(kwargs['greedy_min'])
        self.epsilon_decay = float(kwargs['greedy_decay'])
        self.model = DQN(kwargs)
        self.target_model = DQN(kwargs)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs['learning_rate'])
        self.replay_batch_size = int(kwargs['replay_batch_size'])
        self.algorithm = (kwargs['agent'] if 'agent' in kwargs else 'DQN') 
        self.value_loss_fn = nn.MSELoss(reduction='mean')
        self.temperature_for_action_sampling = float(kwargs['temperature_for_action_sampling'])
        self.device = kwargs['device']


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def remember(self, state, action, reward, next_state, done, step):
        state_to_memorise = state.detach().clone()
        # print(id(state_to_memorise.untyped_storage()))
        next_state_to_memorise = next_state.detach().clone()
        # print(id(next_state_to_memorise.untyped_storage()))
        self.memory[self.memory_position] = (
            state_to_memorise, 
            action, 
            reward, 
            next_state_to_memorise,
            done)
        self.memory_position = (self.memory_position + 1) % self.memory_size
        self.memory_size_actual = min(self.memory_size_actual + 1, self.memory_size)


    def act(self, state):
        with torch.no_grad():
            if self.boltzmann_sampling:
                q_values = self.model(state).squeeze()
                action_probs = torch.softmax(
                    self.temperature_for_action_sampling * q_values,
                    dim=-1).squeeze()
                return torch.multinomial(action_probs, 1).item()
            else:
                if random.random() <= self.epsilon:
                    return random.randrange(self.action_size)
                
                q_values = self.model(state).squeeze()
                return q_values.argmax(0).item()


    def replay(self, step):

        if self.memory_size_actual < self.replay_batch_size:
            return

        indices = random.sample(range(self.memory_size_actual), self.replay_batch_size)
        minibatch = [self.memory[i] for i in indices]

        # Unpack and stack transitions
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # Convert to tensors and move to device
        states      = torch.stack(states).to(self.device)              # shape: [B, state_dim]
        actions     = torch.tensor(actions, dtype=torch.long, device=self.device)  # shape: [B]
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # shape: [B]
        next_states = torch.stack(next_states).to(self.device)         # shape: [B, state_dim]
        dones       = torch.tensor(dones, dtype=torch.bool, device=self.device)     # shape: [B]

        # Compute Q-values for current states using online model
        q_values = self.model(states)                                  # shape: [B, action_dim]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) # shape: [B]

        # Compute target Q-values
        with torch.no_grad():
            if self.algorithm == 'DQN':
                # Use target network to get max Q-values of next states
                next_q_values = self.target_model(next_states).max(1)[0]  # shape: [B]
            elif self.algorithm == 'DDQN':
                # Action selection from online model
                next_actions = self.model(next_states).max(1)[1]          # shape: [B]
                # Evaluation from target model
                next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            # Zero-out next Q-values for terminal states
            next_q_values[dones] = 0.0

            # Bellman target
            target_q_values = rewards + self.gamma * next_q_values  # shape: [B]

        # Compute loss
        loss = self.value_loss_fn(q_values, target_q_values)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log
        if self.wbl: 
            self.wbl.log({
                'active_inference/value_loss': loss.item(),
                'active_inference/pragmatic_gain': rewards.mean().item()
            }, step=step)
            
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
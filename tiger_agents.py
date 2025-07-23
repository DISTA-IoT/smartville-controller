from smartController.neural_modules import DQN, PolicyNet, NEFENet, TransitionNet, VariationalTransitionNet
import torch.optim as optim
from collections import deque
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


class FullDAIAgent:
    def __init__(self, kwargs):
        
        self.wbl = kwargs['wbl']
        self.action_size = kwargs['action_size']
        self.neg_efe_net = NEFENet(kwargs)
        self.target_neg_efe_net = NEFENet(kwargs)
        self.update_target_model()
        self.efe_net_optimizer = optim.Adam(self.neg_efe_net.parameters(), lr=kwargs['learning_rate'])
        self.epistemic_regularisation_factor = kwargs['epistemic_regularisation_factor']
        
        self.transitionnet = None
        self.transitionnet_optimizer = None
        self.variational_t_model = kwargs['variational_tmodel']

        if kwargs['use_transition_model']:

            state_size = kwargs['state_size']
            hidden_state_size = kwargs['h_dim'] + (int(kwargs['use_packet_feats']) * kwargs['h_dim']) + (int(kwargs['node_features']) * kwargs['h_dim'])
            self.proprioceptive_state_size = state_size - hidden_state_size
            kwargs['proprioceptive_state_size'] = self.proprioceptive_state_size

            if self.variational_t_model:
                self.transitionnet = VariationalTransitionNet(kwargs)
                self.variational_variational_transition_loss = kwargs['variational_variational_transition_loss']
                self.kl_divergence_regularisation_factor = kwargs['transitionnet_kl_divergence_regularisation_factor']
            else:
                self.transitionnet = TransitionNet(kwargs)
                
            self.transitionnet_optimizer = optim.Adam(self.transitionnet.parameters(), lr=kwargs['learning_rate'])

        self.policynet = PolicyNet(kwargs)
        self.policynet_optimizer = optim.Adam(self.policynet.parameters(), lr=kwargs['learning_rate'])
        self.temperature_for_action_sampling = kwargs['temperature_for_action_sampling']
        self.entropy_reg_coefficient = kwargs['entropy_reg_coefficient']

        self.memory_size = kwargs['agent_memory_size']
        self.memory = deque(maxlen=self.memory_size)
        self.sequential_memory_size = kwargs['actor_train_interval_steps']
        self.reset_sequential_memory()
        self.replay_batch_size = kwargs['replay_batch_size']

        self.value_loss_fn = nn.MSELoss(reduction='sum')
        self.state_loss_fn = nn.MSELoss(reduction='sum')
        self.surrogate_policy_consistency = kwargs['surrogate_policy_consistency']
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
        self.memory.append((
            state_to_memorise, 
            action, 
            reward, 
            next_state_to_memorise,
            done))
        self.sequential_memory.append(state_to_memorise)
        if len(self.sequential_memory) == self.sequential_memory_size:
            self.train_actor(step)

    
    def act(self, state):
        with torch.no_grad():
            """
            action_probs = self.policynet(state).squeeze()
            """
            if self.use_critic_to_act:
                neg_efe = self.neg_efe_net(state)
                log_action_probs = torch.log_softmax(
                    self.temperature_for_action_sampling * neg_efe,
                    dim=-1).squeeze()
                action_probs = log_action_probs.exp()
            
                # sample from a categorical distribution 
                m = distributions.Categorical(action_probs)
                action = m.sample().item()
            else:
                action = self.policynet(state).argmax().item()

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
        if len(self.memory) < self.replay_batch_size:
            return

        self.neg_efe_net.eval()
        self.target_neg_efe_net.eval()
        

        # Unpack minibatch
        minibatch = random.sample(self.memory, self.replay_batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        action_onehots = torch.nn.functional.one_hot(actions, self.action_size).float()
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
        proprioceptive_states = states[:, -self.proprioceptive_state_size:]
        next_proprioceptive_states = next_states[:, -self.proprioceptive_state_size:]
        transition_inputs = torch.cat([proprioceptive_states, action_onehots], dim=1)
            
        targets = rewards.clone()
        active_epistemic_gains = torch.zeros_like(rewards)
        
        # Vectorized computation of active epistemic gain
        with torch.no_grad():
            action_probs_prior =self.policynet(states)   # [B, A]

            # Compute transition log-likelihoods for ALL actions
            action_onehots_all = torch.eye(self.action_size)  # [A, A]
            expanded_actions = action_onehots_all.repeat(self.replay_batch_size, 1)  # [B*A, A]
            expanded_states = proprioceptive_states.repeat_interleave(self.action_size, dim=0)  # [B*A, S]
            expanded_transition_inputs = torch.cat([expanded_states, expanded_actions], dim=1)
            predicted_nexts = self.transitionnet(expanded_transition_inputs) # [B*A, S']
        
            # Compute log P(o_next | o_current, a)
            log_likelihoods = -F.mse_loss(
                predicted_nexts, 
                next_proprioceptive_states.repeat_interleave(self.action_size, dim=0),
                reduction='none'
            ).sum(dim=1).view(self.replay_batch_size, self.action_size)  # [B, A]
            
            # Bayes' rule: Q(a|s,s') ∝ P(s'|s,a) * Q(a|s)
            log_posterior = log_likelihoods + torch.log(action_probs_prior + 1e-8)
            action_probs_posterior = torch.softmax(log_posterior, dim=1)

            # KL divergence: Σ posterior * log(posterior/prior)
            kl_div = (action_probs_posterior * 
                    (torch.log(action_probs_posterior + 1e-8) - 
                    torch.log(action_probs_prior + 1e-8))
                    ).sum(dim=1, keepdim=True)  # [B, 1]
            
            # active epistemic gain
            active_epistemic_gains = kl_div

        targets += self.epistemic_regularisation_factor * active_epistemic_gains

        
        perceptive_epistemic_gains = torch.zeros_like(rewards)
        # Vectorized computation of perceptive epistemic gain
        with torch.no_grad():
            # These lines approximate the epistemic gain term: \int Q(s)[logQ(s_t) + logQ(s_t|a_t, s_{t-1})]
            # estimated_next_proprioceptive_states <- Q(s_t|a_t, s_{t-1})  {is a  reparameterisation in the variational setting} This is the "variational posterior's prior"
            # next_proprioceptive_states <- Q(s) {is interpreted as a sample from a spherical Gaussian centred on s in the variational setting}. This is the "variational posterior's posterior"
            if self.variational_t_model:
                _, eps_means, eps_logvars = self.transitionnet(transition_inputs)
                # Analytical KL divergence.
                perceptive_epistemic_gains = 0.5 * torch.sum(
                    (1 / torch.exp(eps_logvars)) + ((next_proprioceptive_states - eps_means) ** 2) / torch.exp(eps_logvars)
                    - 1 - eps_logvars,
                    dim=1,
                    keepdim=True
                )
            else:
                estimated_next_proprioceptive_states = self.transitionnet(transition_inputs)
                perceptive_epistemic_gains = torch.sum(
                        (next_proprioceptive_states - estimated_next_proprioceptive_states) ** 2,
                        dim=1,
                        keepdim=True)

            targets += self.epistemic_regularisation_factor * perceptive_epistemic_gains.detach()


        with torch.no_grad():
            # Action selection from online model
            estimated_neg_efe_values = self.neg_efe_net(states).detach()
            efe_actions = torch.log_softmax(
                self.temperature_for_action_sampling * estimated_neg_efe_values, dim=1).max(1)[1] 
            next_efe_values = self.target_neg_efe_net(next_states).gather(1, efe_actions.unsqueeze(1))

            targets -=(~dones) * 0.99 * next_efe_values

            
        # Prepare targets for all actions
        target_neg_efes = self.neg_efe_net(states).detach()
        target_neg_efes[range(self.replay_batch_size), actions] = targets.squeeze()
                
        # train the EFE value network (critic)
        self.neg_efe_net.train()
        predicted_values = self.neg_efe_net(states)
        value_loss = self.value_loss_fn(predicted_values, target_neg_efes)
        self.efe_net_optimizer.zero_grad()
        value_loss.backward()
        self.efe_net_optimizer.step()
    

        # perceptive and policy model training through VFE:
        self.neg_efe_net.eval()
        self.policynet.train()
        self.transitionnet.train()
        vfe = 0

        # The following corresponds Q(a_t | s_t) in eq. (6) 
        policy_probabilities = self.policynet(states) 

        if self.surrogate_policy_consistency:
            policy_consistency = -0.5 * ((policy_probabilities - action_onehots) ** 2).sum(dim=1).mean()
        else:
            # The following 2 loc's correspond p(a|s) according to eq. (8) in the same paper (Boltzman sampling)
            # i.e.: p(a|s) = \sigma(- \gamma G(s,a))
            estimated_neg_efe_values = self.neg_efe_net(states).detach()
            efe_actions = torch.log_softmax(
                self.temperature_for_action_sampling * estimated_neg_efe_values, dim=1)

            # The following 2 loc's correspond to the first term in eq (7), i.e.:
            # -E_{Q(s)}[ \int Q(a|s) logp(a|s) da] 
            # This is the negative of the energy, i.e. the consitency of Q w.r.t p.
            # We need to maximise this energy by minimising VFE which is the negative of this fella.
            policy_consistency = torch.sum(policy_probabilities * efe_actions, dim=1).mean()
                    

        # The following 2 loc's correspond to the second term in eq (7), i.e.:
        # -E_{Q(s)}\{ H[Q(a|s)] \}
        # Also here, we want to maximise the entropy, that's why substract it from the loss.
        policy_log_probs = torch.log(torch.clamp(policy_probabilities, min=1e-8))
        policy_entropy = torch.sum(policy_probabilities * policy_log_probs, dim=1).mean()
        actor_loss = -policy_consistency - self.entropy_reg_coefficient * policy_entropy
        
        # perceptive model
        predicted_observations = self.transitionnet(transition_inputs)
        # Perception consistency (cross entropy ≈ −MSE/2)
        perceptive_consistency = -0.5 * ((next_proprioceptive_states - predicted_observations) ** 2).sum(dim=1).mean()

        # Perception neutrality (entropy proxy using batch variance)
        batch_var = predicted_observations.var(dim=0) + 1e-6
        perceptive_entropy = 0.5 * torch.sum(torch.log(batch_var)) + 0.5 * predicted_observations.shape[1] * torch.log(torch.tensor(2 * torch.pi * torch.e))
        perceptive_loss = -perceptive_entropy * self.entropy_reg_coefficient - perceptive_consistency

        vfe = actor_loss + perceptive_loss

        self.policynet_optimizer.zero_grad()
        self.transitionnet_optimizer.zero_grad()
        vfe.backward()
        self.transitionnet_optimizer.step()
        self.policynet_optimizer.step()
        

        if self.wbl: 
            self.wbl.log({'value_loss': value_loss.item()}, step=step)
            self.wbl.log({'pragmatic_gain': rewards.mean().item()}, step=step) 
            self.wbl.log({'epistemic_gain': perceptive_epistemic_gains.mean().item()}, step=step)
            self.wbl.log({'active_epistemic_gain': active_epistemic_gains.mean().item()}, step=step)
            self.wbl.log({'actor_loss': actor_loss.item()}, step=step)
            self.wbl.log({'perceptive_loss': perceptive_loss.item()}, step=step)
            self.wbl.log({'perceptive_entropy': perceptive_entropy.item()}, step=step) # maximise this (it is positive)
            self.wbl.log({'perceptive_consistency': perceptive_consistency.item()}, step=step) # maximise this (it is positive)
            self.wbl.log({'actor_entropy': policy_entropy.item()}, step=step) # maximise this (it is positive)
            self.wbl.log({'actor_performance': policy_consistency.mean().item()}, step=step) # maximise this (it is positive)
            self.wbl.log({'vfe': vfe.item()}, step=step)


class DAIAgent:
    def __init__(self, kwargs):
        
        self.wbl = kwargs['wbl']
        self.action_size = kwargs['action_size']
        self.neg_efe_net = NEFENet(kwargs)
        self.target_neg_efe_net = NEFENet(kwargs)
        self.update_target_model()
        self.efe_net_optimizer = optim.Adam(self.neg_efe_net.parameters(), lr=kwargs['learning_rate'])
        self.epistemic_regularisation_factor = kwargs['epistemic_regularisation_factor']
        
        self.transitionnet = None
        self.transitionnet_optimizer = None
        self.variational_t_model = kwargs['variational_tmodel']

        if kwargs['use_transition_model']:

            state_size = kwargs['state_size']
            hidden_state_size = kwargs['h_dim'] + (int(kwargs['use_packet_feats']) * kwargs['h_dim']) + (int(kwargs['node_features']) * kwargs['h_dim'])
            self.proprioceptive_state_size = state_size - hidden_state_size
            kwargs['proprioceptive_state_size'] = self.proprioceptive_state_size

            if self.variational_t_model:
                self.transitionnet = VariationalTransitionNet(kwargs)
                self.variational_variational_transition_loss = kwargs['variational_variational_transition_loss']
                self.kl_divergence_regularisation_factor = kwargs['transitionnet_kl_divergence_regularisation_factor']
            else:
                self.transitionnet = TransitionNet(kwargs)
                
            self.transitionnet_optimizer = optim.Adam(self.transitionnet.parameters(), lr=kwargs['learning_rate'])

        self.policynet = PolicyNet(kwargs)
        self.policynet_optimizer = optim.Adam(self.policynet.parameters(), lr=kwargs['learning_rate'])
        self.temperature_for_action_sampling = kwargs['temperature_for_action_sampling']
        self.entropy_reg_coefficient = kwargs['entropy_reg_coefficient']

        self.memory_size = kwargs['agent_memory_size']
        self.memory = deque(maxlen=self.memory_size)
        self.sequential_memory_size = kwargs['actor_train_interval_steps']
        self.reset_sequential_memory()
        self.replay_batch_size = kwargs['replay_batch_size']

        self.value_loss_fn = nn.MSELoss(reduction='sum')
        self.state_loss_fn = nn.MSELoss(reduction='sum')
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
        self.memory.append((
            state_to_memorise, 
            action, 
            reward, 
            next_state_to_memorise,
            done))
        self.sequential_memory.append(state_to_memorise)
        if len(self.sequential_memory) == self.sequential_memory_size:
            self.train_actor(step)

    
    def act(self, state):
        with torch.no_grad():
            """
            action_probs = self.policynet(state).squeeze()
            """
            if self.use_critic_to_act:
                neg_efe = self.neg_efe_net(state)
                log_action_probs = torch.log_softmax(
                    self.temperature_for_action_sampling * neg_efe,
                    dim=-1).squeeze()
                action_probs = log_action_probs.exp()
            
                # sample from a categorical distribution 
                m = distributions.Categorical(action_probs)
                action = m.sample().item()
            else:
                # need to sample treating action probs as categorical, 
                # cuz this agent is on-policy...
                action_probs = self.policynet(state)
                m = distributions.Categorical(action_probs)
                action = m.sample().item()

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

        self.neg_efe_net.eval()

        vfe = 0
        # batching the states
        states = torch.vstack(list(self.sequential_memory))
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
        policy_entropy = torch.sum(policy_probabilities * policy_log_probs, dim=1)
        expected_policy_entropy = policy_entropy.mean()
        vfe -= self.entropy_reg_coefficient * expected_policy_entropy

        self.policynet_optimizer.zero_grad()
        vfe.backward()
        self.policynet_optimizer.step()
        self.reset_sequential_memory()

        if self.wbl: self.wbl.log({'actor_loss': vfe.item()}, step=step)
        if self.wbl: self.wbl.log({'actor_entropy': expected_policy_entropy.item()}, step=step) # maximise this (it is positive)
        if self.wbl: self.wbl.log({'actor_performance': energies.mean().item()}, step=step) # maximise this (it is positive)


    def replay(self, step):
        """
        Use temporal difference on expected free energy to update the critic (efe bootstrapped network)
        This update is based on equation (17) of the Millidge's paper (after the sign correction), which in our paper is:
        \hat{G(s_t,a_t)} =  -r(o) -  \int Q(s)[logQ(s_t) - logQ(s_t|a_t, s_{t-1})] + G_\phi(s_t,a_t)
        which is equivalent to:
        -\hat{G(s_t,a_t)} =  r(o) +  \int Q(s)[logQ(s_t) - logQ(s_t|a_t, s_{t-1})] - G_\phi(s_t,a_t)
        This new form is more of a "value" (a policy's value is inversely prop. to the expected free energy)
        """
        if len(self.memory) < self.replay_batch_size:
            return

        self.neg_efe_net.eval()
        self.target_neg_efe_net.eval()
        if self.transitionnet is not None: self.transitionnet.eval()

        # Unpack minibatch
        minibatch = random.sample(self.memory, self.replay_batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        action_onehots = torch.nn.functional.one_hot(actions, self.action_size).float()
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
        proprioceptive_states = states[:, -self.proprioceptive_state_size:]
        next_proprioceptive_states = next_states[:, -self.proprioceptive_state_size:]
            

        targets = rewards.clone()
        epistemic_gains = torch.zeros_like(rewards)
        
        # Vectorized computation of perceptive epistemic gain
        if self.transitionnet is not None:
            transition_inputs = torch.cat([proprioceptive_states, action_onehots], dim=1)
            
            # These lines approximate the epistemic gain term: \int Q(s)[logQ(s_t) + logQ(s_t|a_t, s_{t-1})]
            # estimated_next_proprioceptive_states <- Q(s_t|a_t, s_{t-1})  {is a  reparameterisation in the variational setting} This is the "variational posterior's prior"
            # next_proprioceptive_states <- Q(s) {is interpreted as a sample from a spherical Gaussian centred on s in the variational setting}. This is the "variational posterior's posterior"
            if self.variational_t_model:
                _, eps_means, eps_logvars = self.transitionnet(transition_inputs)
                # Analytical KL divergence.
                epistemic_gains = 0.5 * torch.sum(
                    (1 / torch.exp(eps_logvars)) + ((next_proprioceptive_states - eps_means) ** 2) / torch.exp(eps_logvars)
                    - 1 - eps_logvars,
                    dim=1,
                    keepdim=True
                )
            else:
                estimated_next_proprioceptive_states = self.transitionnet(transition_inputs)
                epistemic_gains = torch.sum(
                        (next_proprioceptive_states - estimated_next_proprioceptive_states) ** 2,
                        dim=1,
                        keepdim=True)

            targets += self.epistemic_regularisation_factor * epistemic_gains.detach()


        with torch.no_grad():
            # Action selection from online model
            estimated_neg_efe_values = self.neg_efe_net(states).detach()
            efe_actions = torch.log_softmax(
                self.temperature_for_action_sampling * estimated_neg_efe_values, dim=1).max(1)[1] 
            next_efe_values = self.target_neg_efe_net(next_states).gather(1, efe_actions.unsqueeze(1))

            targets -=(~dones) * 0.99 * next_efe_values

        # Prepare targets for all actions
        target_neg_efes = self.neg_efe_net(states).detach()
        target_neg_efes[range(self.replay_batch_size), actions] = targets.squeeze()


        # train the EFE value network (critic)
        self.neg_efe_net.train()
        predicted_values = self.neg_efe_net(states)
        value_loss = self.value_loss_fn(predicted_values, target_neg_efes)
        self.efe_net_optimizer.zero_grad()
        value_loss.backward()
        self.efe_net_optimizer.step()
    
        # Train transition model
        if self.transitionnet is not None:
            self.transitionnet.train()
            if self.variational_t_model:
                estimated_next_proprioceptive_states, l_eps_means, l_eps_logvars = self.transitionnet(transition_inputs)

                if self.variational_variational_transition_loss:
                    # Use means for deterministic target comparison
                    reconstruction_loss = self.state_loss_fn(l_eps_means, next_proprioceptive_states)
                    # KL divergence to standard normal
                    kl_div = 0.5 * torch.sum(
                        l_eps_logvars.exp() + l_eps_means**2 - 1. - l_eps_logvars, 
                        dim=1
                    ).mean()
                    transition_loss = reconstruction_loss + self.kl_divergence_regularisation_factor * kl_div
                    if self.wbl: self.wbl.log({'state_reconstruction_loss': reconstruction_loss.item()}, step=step)
                    if self.wbl: self.wbl.log({'state kl_div': kl_div.item()}, step=step)
                else:
                    transition_loss = self.state_loss_fn(estimated_next_proprioceptive_states, next_proprioceptive_states)
            else:
                estimated_next_proprioceptive_states = self.transitionnet(transition_inputs)
                transition_loss = self.state_loss_fn(estimated_next_proprioceptive_states, next_proprioceptive_states)

            self.transitionnet_optimizer.zero_grad()
            transition_loss.backward()
            self.transitionnet_optimizer.step()

            if self.wbl: 
                self.wbl.log({'transition_loss': transition_loss.item()}, step=step)
                self.wbl.log({'epistemic_gain': epistemic_gains.mean().item()}, step=step)

        if self.wbl: 
            self.wbl.log({'value_loss': value_loss.item()}, step=step)
            self.wbl.log({'pragmatic_gain': rewards.mean().item()}, step=step) 

        

class DDAIAgent:
    def __init__(self, kwargs):
        
        self.wbl = kwargs['wbl']
        self.action_size = kwargs['action_size']
        self.neg_efe_net = NEFENet(kwargs)
        self.target_neg_efe_net = NEFENet(kwargs)
        self.update_target_model()
        self.efe_net_optimizer = optim.Adam(self.neg_efe_net.parameters(), lr=kwargs['learning_rate'])
        self.epistemic_regularisation_factor = kwargs['epistemic_regularisation_factor']
        
        self.transitionnet = None
        self.transitionnet_optimizer = None
        self.variational_t_model = kwargs['variational_tmodel']

        if kwargs['use_transition_model']:

            state_size = kwargs['state_size']
            hidden_state_size = kwargs['h_dim'] + (int(kwargs['use_packet_feats']) * kwargs['h_dim']) + (int(kwargs['node_features']) * kwargs['h_dim'])
            self.proprioceptive_state_size = state_size - hidden_state_size
            kwargs['proprioceptive_state_size'] = self.proprioceptive_state_size

            if self.variational_t_model:
                self.transitionnet = VariationalTransitionNet(kwargs)
                self.variational_variational_transition_loss = kwargs['variational_variational_transition_loss']
                self.kl_divergence_regularisation_factor = kwargs['transitionnet_kl_divergence_regularisation_factor']
            else:
                self.transitionnet = TransitionNet(kwargs)
                
            self.transitionnet_optimizer = optim.Adam(self.transitionnet.parameters(), lr=kwargs['learning_rate'])

        self.policynet = PolicyNet(kwargs)
        self.policynet_optimizer = optim.Adam(self.policynet.parameters(), lr=kwargs['learning_rate'])
        self.temperature_for_action_sampling = kwargs['temperature_for_action_sampling']
        self.entropy_reg_coefficient = kwargs['entropy_reg_coefficient']

        self.memory_size = kwargs['agent_memory_size']
        self.memory = deque(maxlen=self.memory_size)
        self.sequential_memory_size = kwargs['actor_train_interval_steps']
        self.reset_sequential_memory()
        self.replay_batch_size = kwargs['replay_batch_size']

        self.value_loss_fn = nn.MSELoss(reduction='sum')
        self.state_loss_fn = nn.MSELoss(reduction='sum')
        self.use_crictic_to_act = kwargs['use_critic_to_act']

    def reset_sequential_memory(self):
        self.sequential_memory = deque(maxlen=self.sequential_memory_size)

    def update_target_model(self):
        self.target_neg_efe_net.load_state_dict(self.neg_efe_net.state_dict())


    def remember(self, state, action, reward, next_state, done, step):
        state_to_memorise = state.detach().clone()
        # print(id(state_to_memorise.untyped_storage()))
        next_state_to_memorise = next_state.detach().clone()
        # print(id(next_state_to_memorise.untyped_storage()))
        self.memory.append((
            state_to_memorise, 
            action, 
            reward, 
            next_state_to_memorise,
            done))
        self.sequential_memory.append(state_to_memorise)
        if len(self.sequential_memory) == self.sequential_memory_size:
            self.train_actor(step)

    
    def act(self, state):
        with torch.no_grad():
            """
            action_probs = self.policynet(state).squeeze()
            """
            if self.use_crictic_to_act:
                neg_efe = self.neg_efe_net(state)
                log_action_probs = torch.log_softmax(
                    self.temperature_for_action_sampling * neg_efe,
                    dim=-1).squeeze()
                action_probs = log_action_probs.exp()
            
                # sample from a categorical distribution 
                m = distributions.Categorical(action_probs)
                action = m.sample().item()
            else:
                # DDPG STYLE
                action_probs = self.policynet(state).squeeze()
                action = action_probs.max(0)[1].item()

        return action
    

    def train_actor(self, step):
        # this trains not the actor but the perceptvie model
        vfe = 0
        
        # batching the states
        states = torch.vstack(list(self.sequential_memory))
        proprioceptive_states = states[:, -self.proprioceptive_state_size:]
        estimated_neg_efe_values = self.neg_efe_net(states).detach()
        efe_actions = torch.log_softmax(
            self.temperature_for_action_sampling * estimated_neg_efe_values, dim=1).max(dim=1)[1]
        action_onehots = torch.nn.functional.one_hot(efe_actions, self.action_size).float()
        transition_inputs = torch.cat([proprioceptive_states, action_onehots], dim=1)

        self.transitionnet.train()
        predicted_observations = self.transitionnet(transition_inputs)
        
        # Perception consistency (cross entropy ≈ −MSE/2)
        perceptive_consistency = -0.5 * ((proprioceptive_states[1:] - predicted_observations[:-1]) ** 2).sum(dim=1).mean()

        # Perception neutrality (entropy proxy using batch variance)
        batch_var = predicted_observations.var(dim=0) + 1e-6
        perceptive_entropy = 0.5 * torch.sum(torch.log(batch_var)) + 0.5 * predicted_observations.shape[1] * torch.log(torch.tensor(2 * torch.pi * torch.e))

        
        vfe = - perceptive_entropy * self.entropy_reg_coefficient - perceptive_consistency
        if self.wbl: self.wbl.log({'perceptive_entropy': perceptive_entropy.item()}, step=step) # maximise this (it is positive)
        if self.wbl: self.wbl.log({'perceptive_consistency': perceptive_consistency.item()}, step=step) # maximise this (it is positive)

        self.transitionnet_optimizer.zero_grad()
        vfe.backward()
        self.transitionnet_optimizer.step()
        self.reset_sequential_memory()

        if self.wbl: self.wbl.log({'perceptive_loss': vfe.item()}, step=step)


    def replay(self, step):

        if len(self.memory) < self.replay_batch_size:
            return

        self.neg_efe_net.eval()
        self.target_neg_efe_net.eval()
        if self.transitionnet is not None: self.transitionnet.eval()

        # Unpack minibatch
        minibatch = random.sample(self.memory, self.replay_batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        action_onehots = torch.nn.functional.one_hot(actions, self.action_size).float()
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
        proprioceptive_states = states[:, -self.proprioceptive_state_size:]
        next_proprioceptive_states = next_states[:, -self.proprioceptive_state_size:]
            

        targets = rewards.clone()
        active_epistemic_gains = torch.zeros_like(rewards)
        
        # Vectorized computation of active epistemic gain
        with torch.no_grad():
            action_probs_prior =self.policynet(states)   # [B, A]

            # Compute transition log-likelihoods for ALL actions
            action_onehots_all = torch.eye(self.action_size)  # [A, A]
            expanded_actions = action_onehots_all.repeat(self.replay_batch_size, 1)  # [B*A, A]
            expanded_states = proprioceptive_states.repeat_interleave(self.action_size, dim=0)  # [B*A, S]
            transition_inputs = torch.cat([expanded_states, expanded_actions], dim=1)
            predicted_nexts = self.transitionnet(transition_inputs) # [B*A, S']
        
            # Compute log P(o_next | o_current, a)
            log_likelihoods = -F.mse_loss(
                predicted_nexts, 
                next_proprioceptive_states.repeat_interleave(self.action_size, dim=0),
                reduction='none'
            ).sum(dim=1).view(self.replay_batch_size, self.action_size)  # [B, A]
            
            # Bayes' rule: Q(a|s,s') ∝ P(s'|s,a) * Q(a|s)
            log_posterior = log_likelihoods + torch.log(action_probs_prior + 1e-8)
            action_probs_posterior = torch.softmax(log_posterior, dim=1)

            # KL divergence: Σ posterior * log(posterior/prior)
            kl_div = (action_probs_posterior * 
                    (torch.log(action_probs_posterior + 1e-8) - 
                    torch.log(action_probs_prior + 1e-8))
                    ).sum(dim=1, keepdim=True)  # [B, 1]
            
            # active epistemic gain
            active_epistemic_gains = kl_div

        targets += self.epistemic_regularisation_factor * active_epistemic_gains


        with torch.no_grad():
            # Action selection from online model
            estimated_neg_efe_values = self.neg_efe_net(states).detach()
            efe_actions = torch.log_softmax(
                self.temperature_for_action_sampling * estimated_neg_efe_values, dim=1).max(1)[1] 
            next_efe_values = self.target_neg_efe_net(next_states).gather(1, efe_actions.unsqueeze(1))

            targets -=(~dones) * 0.99 * next_efe_values

        # Prepare targets for all actions
        target_neg_efes = self.neg_efe_net(states).detach()
        target_neg_efes[range(self.replay_batch_size), actions] = targets.squeeze()


        # train the EFE value network (critic)
        self.neg_efe_net.train()
        predicted_values = self.neg_efe_net(states)
        value_loss = self.value_loss_fn(predicted_values, target_neg_efes)
        self.efe_net_optimizer.zero_grad()
        value_loss.backward()
        self.efe_net_optimizer.step()
    
        # train the policy network:
        self.policynet.train()
        predicted_actions = self.policynet(states)
        policy_loss = torch.sum((predicted_actions - action_onehots) ** 2)
        self.policynet_optimizer.zero_grad()
        policy_loss.backward()
        self.policynet_optimizer.step()
        
        if self.wbl: 
            self.wbl.log({'policy_loss': policy_loss.item()}, step=step)
            self.wbl.log({'pragmatic_gain': rewards.mean().item()}, step=step) 
            self.wbl.log({'value_loss': value_loss.item()}, step=step)
            

class ValueLearningAgent:

    def __init__(self, kwargs):
        self.wbl = kwargs['wbl']
        self.state_size = kwargs['state_size']
        self.action_size = kwargs['action_size']
        self.memory = deque(maxlen=kwargs['agent_memory_size'])
        self.gamma = float(kwargs['agent_discount_rate'])  # discount rate
        self.epsilon = float(kwargs['init_epsilon_egreedy'])  # exploration rate
        self.epsilon_min = float(kwargs['greedy_min'])
        self.epsilon_decay = float(kwargs['greedy_decay'])
        self.model = DQN(kwargs)
        self.target_model = DQN(kwargs)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs['learning_rate'])
        self.replay_batch_size = kwargs['replay_batch_size']
        self.algorithm = (kwargs['agent'] if 'agent' in kwargs else 'DQN') 
        self.value_loss_fn = nn.MSELoss(reduction='mean')
        self.device = kwargs['device']


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def remember(self, state, action, reward, next_state, done, step):
        state_to_memorise = state.detach().clone()
        # print(id(state_to_memorise.untyped_storage()))
        next_state_to_memorise = next_state.detach().clone()
        # print(id(next_state_to_memorise.untyped_storage()))
        self.memory.append((
            state_to_memorise, 
            action, 
            reward, 
            next_state_to_memorise,
            done))


    def act(self, state):

        if torch.rand(1).item() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model(state).squeeze()
        return q_values.max(0)[1].item()


    def replay(self, step):

        if len(self.memory) < self.replay_batch_size:
            return

        minibatch = random.sample(self.memory, self.replay_batch_size)

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
        self.wbl.log({'value_loss': loss.item()}, step=step)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
from smartController.neural_modules import DQN, PolicyNet, NEFENet, TransitionNet, VariationalTransitionNet
import torch.optim as optim
from collections import deque
import torch
import random
import torch.nn as nn
import torch.distributions as distributions

class DAIAgent:
    def __init__(self, kwargs):
        
        self.wbl = kwargs['wbl']
        self.action_size = kwargs['action_size']
        self.neg_efe_net = NEFENet(kwargs)
        self.target_neg_efe_net = NEFENet(kwargs)
        self.update_target_model()
        self.efe_net_optimizer = optim.Adam(self.neg_efe_net.parameters(), lr=kwargs['learning_rate'])
        
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
            else:
                self.transitionnet = TransitionNet(kwargs)
                
            self.transitionnet_optimizer = optim.Adam(self.transitionnet.parameters(), lr=kwargs['learning_rate'])

        self.policynet = PolicyNet(kwargs)
        self.policynet_optimizer = optim.Adam(self.policynet.parameters(), lr=kwargs['learning_rate'])

        self.memory_size = kwargs['agent_memory_size']
        self.memory = deque(maxlen=self.memory_size)
        self.reset_sequential_memory()
        self.replay_batch_size = kwargs['replay_batch_size']

        self.value_loss_fn = nn.MSELoss(reduction='sum')
        self.state_loss_fn = nn.MSELoss(reduction='sum')

    def reset_sequential_memory(self):
        self.sequential_memory = deque(maxlen=self.memory_size)

    def update_target_model(self):
        self.target_neg_efe_net.load_state_dict(self.neg_efe_net.state_dict())


    def remember(self, state, action, reward, next_state, done):
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

    
    def act(self, state):

        action_probs = self.policynet(state).squeeze()
        
        # sample from a categorical distribution 
        m = distributions.Categorical(action_probs)
        action = m.sample().item()

        return action
    

    def train_actor(self, step):
        """
        Trains the actor (policy network) by minimising the VFE.
        
        NOTE: THIS METHOD SOULD BE DONE ON-POLICY

        With respect to the paper's equation (6) (wich we correct the sign), i.e.:

        VFE =\int Q(s)logp(o|s) + KL(Q(s)||p(s|s_{t-1},a_{t-1})] + E_{Q(s)}[KL[Q(a|s)||p(a|s)]

        - We do not have a POMDP but only an MDP, so we do not minimise accuracy over observations (\int Q(s)logp(o|s) = 1)

        - We are not touching the KL(Q(s)||p(s|s_{t-1},a_{t-1})] term either, because that will be the focus of the critic
        in the context of bootstrapping the EFE. (not sure)

        """
        self.neg_efe_net.eval()

        vfe_loss = 0

        states = torch.vstack(list(self.sequential_memory))
        
        # The following corresponds Q(a_t | s_t) in eq (6) of Millidge's paper (DAI as Variational Policy Gradients)
        policy_probabilities = self.policynet(states) 
        
        # The following 2 loc's correspond to eq (8) (Boltzman sampling)
        # i.e.: p(a|s) = \sigma(- \gamma G(s,a))
        estimated_neg_efe_values = self.neg_efe_net(states).detach()
        efe_actions = torch.log_softmax(estimated_neg_efe_values, dim=1)

        # The last term in paper's eq (6) is E_{Q(s)}[KL[Q(a|s)||p(a|s)]
        # which divides into two terms:

        # The following 2 loc's correspond to the first term in eq (7), i.e.:
        # -E_{Q(s)}[ \int Q(a|s) logp(a|s) da] 
        # This is kind of performance measure, that is why we subtract it 
        # (because we want to MINIMISE the loss function, i.e., maximise the performance)
        expected_logprobs = torch.sum(policy_probabilities * efe_actions, dim=1)
        vfe_loss -= expected_logprobs.mean()

        # The following 2 loc's correspond to the second term in eq (7), i.e.:
        # -E_{Q(s)}\{ H[Q(a|s)] \}
        # Also here, we want to maximise the entropy, that's why substract it from the loss.
        policy_log_probs = torch.log(policy_probabilities + 1e-8)
        policy_entropy = torch.sum(policy_probabilities * policy_log_probs, dim=1)
        expected_policy_entropy = policy_entropy.mean()
        vfe_loss -= expected_policy_entropy

        self.policynet_optimizer.zero_grad()
        vfe_loss.backward()
        self.policynet_optimizer.step()
        self.reset_sequential_memory()

        self.wbl.log({'actor_loss': vfe_loss.item()}, step=step)
        self.wbl.log({'actor_entropy': expected_policy_entropy.item()}, step=step) # maximise this (it is positive)
        self.wbl.log({'actor_performance': expected_logprobs.mean().item()}, step=step) # maximise this (it is positive)


    def replay(self, step):
        """
        Use temporal difference on expected free energy to update the critic (efe bootstrapped network)
        
        This update is based on equation (17) of the paper, i.e.:
        
        \hat{G(s,a)} =  -r(o) +  \int Q(s)[logQ(s) - logQ(s|o)] + G_\phi(s,a)
        
        which is equivalent to:
        
        -\hat{G(s,a)} =  r(o) -  \int Q(s)[logQ(s) + logQ(s|o)] - G_\phi(s,a)

        This new form is more of a "value" (your policy's value is inversely prop. to the expected free energy)
        
        """
        if len(self.memory) < self.replay_batch_size:
            return
        
        minibatch_states = []
        minibatch_proprioceptive_states = []
        minibatch_next_proprioceptive_states = []
        minibatch_actions = []
        minibatch_neg_efes_targets = []
        minibatch_epistemic_losses = []
        minibatch_pragmatic_gains = []

        minibatch = random.sample(self.memory, self.replay_batch_size)

        # print(len(set([id(tuplesita[0].untyped_storage()) for tuplesita in minibatch ])))
        # print(len(set([id(tuplesita[3].untyped_storage()) for tuplesita in minibatch ])))

        
        self.neg_efe_net.eval()
        self.target_neg_efe_net.eval()
        if self.transitionnet is not None: self.transitionnet.eval()

        for state, action, reward, next_state, done in minibatch:   

            # print(id(next_state.untyped_storage()))

            # reward = r(o)
            target = reward

            minibatch_pragmatic_gains.append(reward)

            action_onehot = torch.zeros(self.action_size)
            action_onehot[action] = 1

            
            if self.transitionnet is not None:
                proprioceptive_state = state[-self.proprioceptive_state_size:]
                next_proprioceptive_state = next_state[-self.proprioceptive_state_size:]

            if not done:
                
                # The following three LOC's
                # Compute efe value under the current policy, i.e. they implement G_\phi(s,a)
                # which is a bootstrapping approximation of the last term in equation (16), i.e.:
                # E_{Q(s_{t+1},a_{t+1})}[\sum_{t+1}^TG(s_{t+1},a_{t+1})]
                policy_probabilities = self.policynet(next_state).squeeze() 
                estimated_neg_EFE_values = self.target_neg_efe_net(next_state).squeeze()                
                approx_rest_neg_efe_value = torch.sum(policy_probabilities * estimated_neg_EFE_values, dim=0)

                if self.transitionnet is not None:

                    # if we have a transition network, then we compute the epistemic gain w.r.t to it. 
                    # These lines approximate  -  \int Q(s)[logQ(s) + logQ(s|o)]
                    # estimated_next_state = Q(s|o)
                    if self.variational_t_model:
                        estimated_next_proprioceptive_state, eps_mean, eps_logvar = self.transitionnet(torch.cat([proprioceptive_state, action_onehot]))
                    else:
                        estimated_next_proprioceptive_state = self.transitionnet(torch.cat([proprioceptive_state, action_onehot]))
                                        
                    # approx_epistemic_loss approximates \int Q(s)[logQ(s) + logQ(s|o)] 
                    # state = Q(s), cuz we're on a fully observable MDP (ground truth assumed Dirac δ)
                    if self.variational_t_model:
                        # Compute analytical KL divergence 
                        approx_epistemic_loss = 0.5 * torch.sum(
                            eps_logvar.exp() + (eps_mean - next_proprioceptive_state)**2 - 1 - eps_logvar,
                            dim=1
                        ).mean()
                    else:
                        approx_epistemic_loss = torch.sum((next_proprioceptive_state - estimated_next_proprioceptive_state.detach()) ** 2)

                    minibatch_epistemic_losses.append(approx_epistemic_loss.detach())

                    # we subtrack the KL divergence from the regression target 
                    # ( we want to minimise this divergence)
                    target -= approx_epistemic_loss
                
                # Update target
                target -= 0.99 * approx_rest_neg_efe_value.detach()

            
            target_f = self.neg_efe_net(state).detach().squeeze()
            target_f[action] = target
        
            minibatch_neg_efes_targets.append(target_f)

            s = state.detach().clone()
            minibatch_states.append(s)

            a = action_onehot.detach().clone()
            minibatch_actions.append(a)
            
            if self.transitionnet is not None:

                p = proprioceptive_state.detach().clone()
                minibatch_proprioceptive_states.append(p)

                next_p = next_proprioceptive_state.detach().clone()
                minibatch_next_proprioceptive_states.append(next_p)


        minibatch_states = torch.vstack(minibatch_states)
        minibatch_actions = torch.vstack(minibatch_actions)
        minibatch_neg_efes_targets = torch.vstack(minibatch_neg_efes_targets)
        minibatch_pragmatic_gains = torch.vstack(minibatch_pragmatic_gains)

        if self.transitionnet is not None:
            minibatch_proprioceptive_states = torch.vstack(minibatch_proprioceptive_states)
            minibatch_next_proprioceptive_states = torch.vstack(minibatch_next_proprioceptive_states)
            minibatch_epistemic_losses = torch.vstack(minibatch_epistemic_losses)

            # train the transition network:
            self.transitionnet.train()
            inputs_to_transitionnet = torch.hstack([minibatch_proprioceptive_states, minibatch_actions])
            estimated_next_proprioceptive_states = self.transitionnet(inputs_to_transitionnet)
            self.transitionnet_optimizer.zero_grad()
            transition_loss = self.state_loss_fn(estimated_next_proprioceptive_states, minibatch_next_proprioceptive_states)
            transition_loss.backward()
            self.transitionnet_optimizer.step()

            self.wbl.log({'transition_loss': transition_loss.item()}, step=step)
            self.wbl.log({'epistemic_gain': -minibatch_epistemic_losses.mean().item()}, step=step)


        # train the value network
        self.neg_efe_net.train()
        predicted_values = self.neg_efe_net(minibatch_states)
        self.efe_net_optimizer.zero_grad()
        value_losses = self.value_loss_fn(predicted_values, minibatch_neg_efes_targets)
        value_losses.backward()
        self.efe_net_optimizer.step()
    
        self.wbl.log({'value_loss': value_losses.item()}, step=step)
        self.wbl.log({'pragmatic_gain': minibatch_pragmatic_gains.mean().item()}, step=step) 

        

class ValueLearningAgent:

    def __init__(self, kwargs):

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

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def remember(self, state, action, reward, next_state, done):
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


    def train_actor(self, step):
        # added for compatibility TODO implement polimorfism
        pass


    def act(self, state):

        if torch.rand(1).item() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model(state).squeeze()
        return q_values.max(0)[1].item()


    def replay(self, step):

        if len(self.memory) < self.replay_batch_size:
            return

        minibatch = random.sample(self.memory, self.replay_batch_size)

        # print(len(set([id(tuplesita[0].untyped_storage()) for tuplesita in minibatch ])))
        # print(len(set([id(tuplesita[3].untyped_storage()) for tuplesita in minibatch ])))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            
            if not done:

                if self.algorithm == 'DQN':  
                    # take the max Q-value from target network
                    target += self.gamma * torch.max(self.target_model(next_state).squeeze()).item()
                elif self.algorithm == 'DDQN':
                    # Select action using online network
                    next_action = self.model(next_state).squeeze().max(0)[1].item()
                    # Evaluate using target network
                    target += self.gamma * self.target_model(next_state).squeeze()[next_action].item()  
        
            
            target_f = self.model(state).detach().squeeze()
            target_f[action] = target


            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state).squeeze(), target_f)
            loss.backward()
            self.optimizer.step()

            self.wbl.log({'value_loss': loss.item()}, step=step)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
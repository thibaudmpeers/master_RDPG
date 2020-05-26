import torch
from torch.nn.functional import smooth_l1_loss


class TBTT_critic:
    def __init__(self, critic_model, truncated_param, optimizer, number_steps_traj, gamma, writer):
        self.critic_model = critic_model
        self.T = truncated_param
        self.optimizer = optimizer
        self.steps_traj = number_steps_traj
        self.gamma = gamma
        self.writer = writer

        self.count = 0

    def train(self, batch_histoy):
        observations_hist, actions_hist, rewards_hist, inputs_hist, target_values_hist, count = batch_histoy
        init_state = self.critic_model.init_hidden_state(count)
        states = [(None, init_state)]

        inputs_hist.requires_grad = True
        actions_hist.requires_grad = True

        critic_losses = torch.zeros(1)

        mean_errors = torch.zeros(1)
        mean_relative_errors = torch.zeros(1)
        last_bias = torch.zeros(self.steps_traj)
        grad_last_bias = torch.zeros(self.steps_traj)

        mean_grad_input_2 = torch.zeros(self.steps_traj)
        mean_grad_input_0 = torch.zeros(self.steps_traj)

        for j in range(self.steps_traj):
            inputs = inputs_hist[:, j].detach()
            target_value = target_values_hist[:, j].detach()
            reward = rewards_hist[:, j].detach()
            action = actions_hist[:, j].detach()

            target_critic = reward + self.gamma * target_value

            state = states[-1][1].detach()
            state.requires_grad = True
            value, next_state = self.critic_model(inputs, action, state)
            states.append((state, next_state))

            while len(states) > self.T:
                del states[0]

            critic_loss = (value - target_critic).pow(2).mean()

            with torch.no_grad():
                mean_error = (value - target_critic).abs().mean()
                mean_errors += (mean_error / self.steps_traj)
                mean_relative_error = ((value - target_critic)/target_critic).abs().mean()
                mean_relative_errors += mean_relative_error / self.steps_traj
                critic_losses += critic_loss

            self.optimizer.zero_grad()

            critic_loss.backward(retain_graph=True)

            # grad_on_previous_reward += states[-1][0][0].grad

            for i in range(self.T - 1):
                if states[-i - 2][0] is None:
                    break
                current_grad = states[-i - 1][0].grad
                states[-i - 2][1].backward(current_grad, retain_graph=True)

            last_bias[j] = self.critic_model.output_layer.bias.data
            grad_last_bias[j] = self.critic_model.output_layer.bias.grad.data
            mean_grad_input_2[j] = self.critic_model.rnn_layer.weight_ih.data.grad[:, -2:].mean() / self.steps_traj
            mean_grad_input_0[j] = self.critic_model.rnn_layer.weight_ih.data.grad[:, :-2].mean() / self.steps_traj

            self.optimizer.step()

        self.writer.add_scalar('mean_relative_errors', mean_relative_errors, self.count)
        self.writer.add_scalar('mean_errors', mean_errors, self.count)
        self.writer.add_scalar('critic_loss', critic_losses, self.count)
        self.writer.add_histogram('last_bias', last_bias, self.count)
        self.writer.add_histogram('grad_last_bias', grad_last_bias, self.count)
        self.writer.add_histogram('inputs_2_weight', self.critic_model.rnn_layer.weight_ih.data[:, -2:], self.count)
        self.writer.add_histogram('inputs_2_weight_grad_critic', mean_grad_input_2, self.count)
        self.writer.add_histogram('inputs_0_weight', self.critic_model.rnn_layer.weight_ih.data[:, :-2], self.count)
        self.writer.add_histogram('inputs_0_weight_grad_critic', mean_grad_input_0, self.count)
        self.count += 1

        return critic_losses.cpu().numpy(), mean_errors, mean_relative_errors


class TBTT_actor:
    def __init__(self, actor_model, T, optimizer, number_steps_traj, critic_model, writer):
        self.actor_model = actor_model
        self.T = T
        self.optimizer = optimizer
        self.steps_traj = number_steps_traj
        self.critic_model = critic_model
        self.writer = writer

        self.count = 0

    def train(self, batch_history):
        observations_hist, actions_hist, rewards_hist, inputs_hist, target_values_hist, count = batch_history

        init_actor_state = self.actor_model.init_hidden_state(count).detach()
        init_critic_state = self.critic_model.init_hidden_state(count).detach()

        states = [(None, init_actor_state)]
        critic_state = init_critic_state

        inputs_hist.requires_grad = True

        actor_losses = torch.zeros(1)

        mean_grad_input_2 = torch.zeros(self.steps_traj)
        mean_grad_input_0 = torch.zeros(self.steps_traj)

        for j in range(self.steps_traj):

            inputs = inputs_hist[:, j]

            state = states[-1][1].detach()
            state.requires_grad = True
            action, next_state = self.actor_model(inputs, state)
            states.append((state, next_state))
            value, critic_state = self.critic_model(inputs, action, critic_state.detach())

            actor_loss = -value.mean()

            with torch.no_grad():
                actor_losses += actor_loss

            self.optimizer.zero_grad()

            actor_loss.backward(retain_graph=True)

            for i in range(self.T - 1):
                if states[-i - 2][0] is None:
                    break
                current_grad = states[-i - 1][0].grad
                states[-i - 2][1].backward(current_grad, retain_graph=True)

            mean_grad_input_2[j] = self.actor_model.rnn_layer.weight_ih.data.grad[:, -2:].mean() / self.steps_traj
            mean_grad_input_0[j] = self.actor_model.rnn_layer.weight_ih.data.grad[:, :-2].mean() / self.steps_traj

            self.optimizer.step()

        self.writer.add_histogram('inputs_2_weight_grad_actor', mean_grad_input_2, self.count)
        self.writer.add_histogram('inputs_0_weight_grad_actor', mean_grad_input_0, self.count)
        self.count += 1

        return actor_losses.cpu().numpy()

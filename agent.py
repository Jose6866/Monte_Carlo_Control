import numpy as np
from visualize_train import draw_value_image, draw_policy_image

# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]

TRAINING_EPISODE_NUM = 100

class AGENT:
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = env.size()
        self.state = [0,0]

        if is_upload:   # Test
            mcc_results = np.load('./result/mcc.npz')
            self.V_values = mcc_results['V']
            self.Q_values = mcc_results['Q']
            self.policy = mcc_results['PI']
        else:          # For training
            self.V_values = np.zeros((HEIGHT, WIDTH))
            self.Q_values = np.zeros((HEIGHT, WIDTH, len(self.ACTIONS)))
            self.policy = np.zeros((HEIGHT, WIDTH, len(self.ACTIONS)))+1./len(self.ACTIONS)



    def initialize_episode(self):
        HEIGHT, WIDTH = self.env.size()
        while True:
            i = np.random.randint(HEIGHT)
            j = np.random.randint(WIDTH)
            state = [i, j]
            if (state in self.env.goal) or (state in self.env.obstacles):
                continue
            break
            # if (state not in self.env.goal) and (state not in self.env.obstacles):
            #     break
        return state



    def Monte_Carlo_Control(self, discount=1.0, alpha=0.01, max_seq_len=500,
                            epsilon=0.3, decay_period=20000, decay_rate=0.9):

        step = 0
        for episode in range(TRAINING_EPISODE_NUM):
            state = self.initialize_episode()

            done = False
            timeout = False
            seq_len = 0
            history = []


            # Sequence generation
            while not self.env.is_terminal(state): # terminal에 도착할 때 까지 episode 생성
                action = self.get_action(state) # policy를 따라 해당 state의 action 생성
                next_state, reward = self.env.interaction(state, ACTIONS[action]) # 해당 state에서 뽑은 action을 하였을때 결과
                history.append((state, action, next_state, reward)) # 한 episode에 대해 각 state에서의 결과를 저장
                state = next_state # state를 next_state로 이동
                seq_len +=1 # episode가 너무 길어지면 빠져나옴
                if seq_len == max_seq_len:
                    break

            # Q Value and policy update
            cum_reward = 0 # G_t
            state_history=[] # first-visit을 위해 state와 action pair을 저장
            for ([i,j], action, [i_next, j_next], reward) in reversed(history): # 각 state에 대해 계산
                cum_reward = reward + (discount * cum_reward)  # G-t update
                if not ([i,j], action) in state_history: # first-visit인 경우인지 확인
                    state_history.append(([i,j], action)) # first-visit인 경우만 저장
                    self.Q_values[i][j][action] += alpha * (cum_reward - self.Q_values[i][j][action]) # Q(S,A) update

                    new_action_direction = np.argmax(self.Q_values[i][j]) # 새로운 action의 방향(index) 저장
                    old_action_direction = np.argmax(self.policy[i][j])  # 기존의 action의 방향(index) 저장

                    if new_action_direction == old_action_direction: # 기존 policy가 새로운 policy와 같은 경우 다음 state로 넘어감
                        if self.policy[i][j][new_action_direction] == 1 - epsilon + epsilon/len(self.ACTIONS):
                            continue

                    for t in range (len(self.policy[i][j])): # policy를 update
                        if t == new_action_direction: # greedy한 방향인 경우
                            self.policy[i][j][t] = 1 - epsilon + epsilon/len(self.ACTIONS)
                        else: # greedy한 방향이 아닌 경우
                            self.policy[i][j][t] = epsilon/len(self.ACTIONS)

            if step % decay_period == 0: # epsilon을 2000episode마다 줄여주어 수렴이 잘 되도록 함
                epsilon = epsilon * decay_rate

            step +=1
            print('step {}:'.format(step))


        self.V_values = np.max(self.Q_values, axis=2)
        draw_value_image(1, np.round(self.V_values, decimals=2), env=self.env)
        draw_policy_image(1, np.round(self.policy, decimals=2), env=self.env)
        np.savez('./result/mcc.npz', Q=self.Q_values, V=self.V_values, PI=self.policy)
        return self.Q_values, self.V_values, self.policy



    def get_action(self, state):
        i,j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i,j,:].tolist()).item()


    def get_state(self):
        return self.state


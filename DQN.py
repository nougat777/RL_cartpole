import gym
import gym.envs
import torch.nn as nn
import torch
import numpy as np

class Mymodel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 40)
        self.fc2 = nn.Linear(40, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=object)
        self.index = 0
        self.size = 0
    def push(self, state, action, reward, next_state):
        self.buffer[self.index] =  state, action, reward, next_state
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        batch_size = min(self.size,batch_size)
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = np.array(self.buffer[indices], dtype=object)
        state, action, reward, next_state = map(np.stack, zip(*batch)) 
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        return state, action, reward, next_state

modelpth_name = 'mynet.pth'
gamma = .95
learning_rate = .001
memory_size = 10000
batch_size = 50
exploration_init = .6
exploration_min = 0.01
exploration_decay = 0.99
train_max_episode = 1000
train_max_step = 10000
test_max_episode = 500
test_max_step = 500
test_truncated_target = test_max_episode *.95
device = 'cuda' if torch.cuda.is_available() else 'cpu'
replaybuffer = ReplayBuffer(memory_size)
policynet = Mymodel().to(device)
targetnet = Mymodel().to(device)
optimizer = torch.optim.Adam(policynet.parameters(), lr=learning_rate,amsgrad=True)
def get_reward(state:object,done:bool,truncated:bool):
    reward = 1
    if done:
        reward = -1
    return reward
def data_normalize(state):
    data_min,data_max = -2.5,2.5
    result = np.zeros(len(state))
    for i in range(len(state)):
        result[i] = (state[i] - data_min) / (data_max - data_min)
    return result  
def update_model(targetnet:nn.Module,replaybuffer:ReplayBuffer):
    loss_function = nn.MSELoss()
    state,action,reward,next_state = replaybuffer.sample(batch_size)
    policynet.train()
    q_value = policynet(state).gather(1,action.unsqueeze(-1))
    with torch.no_grad():
        next_q_value = targetnet(next_state).max(1)[0].unsqueeze(-1)
        reward = reward.unsqueeze(-1)
        expected_q_value = reward + gamma * next_q_value
    loss = loss_function(q_value,expected_q_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    policynet.eval()
    return loss.item()
def train_model():
    #====INIT BEGIN====
    exploration = exploration_init
    targetnet.load_state_dict(policynet.state_dict())
    targetnet.eval()
    policynet.eval()
    env = gym.make('CartPole-v1',render_mode='rgb_array')
    best_state_dict = None
    max_rewards = 0
    rewardss = 0
    episode = 0
    #==== INIT END ====
    while True:
        rewards = 0
        state,info = env.reset()
        state = data_normalize(state)
        for _ in range(train_max_step):
            state_tensor = torch.tensor(state,dtype=torch.float32).to(device)
            q_value = policynet(state_tensor)
            action = np.random.choice(2) if np.random.random() < exploration else torch.argmax(q_value).item()
            next_state,reward,done,truncated,info = env.step(action)
            reward = get_reward(next_state,done,truncated)
            next_state = data_normalize(next_state)
            replaybuffer.push(state,action,reward,next_state)
            state = next_state
            rewards+=reward
            if done:
                break
        rewardss += rewards
        if max_rewards < rewards:
            max_rewards = rewards
            torch.save(policynet.state_dict(),'./'+modelpth_name)           
            best_state_dict = policynet.state_dict()
        print(f'\r(TRAIN) {(episode)/train_max_episode*100:6.1f}% avgR:{rewardss/(episode+1):4.0f} maxR:{max_rewards:4.0f}',end='',flush=True)
        exploration = max(exploration_min,exploration*exploration_decay)
        if episode%2 == 0:
            targetnet.load_state_dict(policynet.state_dict())
        if episode >= train_max_episode:
            break
        else:
            episode +=1
        if rewardss/(episode+1) < 100 and episode > train_max_episode *.5:
            episode = 0
            rewardss = 0
    env.close()
    print()
    return best_state_dict
def test_model(state_dict = None):
    testnet = Mymodel().to(device)
    if state_dict == None:
        try:
            testnet.load_state_dict(torch.load('./'+modelpth_name,weights_only=True))    
        except Exception as e:
            print(e)
            return False
    else:
        testnet.load_state_dict(state_dict)
    env = gym.make('CartPole-v1',render_mode='rgb_array')
    truncated_flag = False
    truncated_count = 0
    max_rewards = 0
    rewardss = 0
    for episode in range(test_max_episode):
        state,info = env.reset()
        state = data_normalize(state)
        rewards = 0
        for _ in range(test_max_step):
            state_tensor = torch.tensor(state,dtype=torch.float32).to(device)
            q_values = testnet(state_tensor)
            action = torch.argmax(q_values).item()
            next_state,reward,done,truncated,info = env.step(action)
            reward = get_reward(next_state,done,truncated)
            state = next_state
            state = data_normalize(state)
            rewards += reward
            if truncated:
                truncated_flag = True
            if rewards > 10000:
                break
            if done:
                break
        if truncated_flag:
            truncated_count += 1
            truncated_flag = False
        rewardss += rewards
        max_rewards = max(rewards,max_rewards)
        print(f'\r( TEST) {(episode)/test_max_episode*100:6.1f}% avgR:{rewardss / (episode+1):4.0f} maxR:{max_rewards:4.0f} truncated:{truncated_count:3d}',end='',flush=True)
        if truncated_count + test_max_episode - episode < test_truncated_target:
            print('\033[4 early quit\033[0m')
            break
    env.close()
    print()
    if truncated_count >= test_truncated_target:
        return True
    return False
def main():
    state_dict = None
    try:
        policynet.load_state_dict(torch.load('./'+modelpth_name,weights_only=True))
    except Exception as e:
        print(e)
        pass
    try:
        while(1):
            state_dict = train_model()
            if(test_model(state_dict)):
                print('sucess!')
                break
    except KeyboardInterrupt:
        print("\r\nInterrput")
if __name__ == '__main__':
    main()
from environment import grid_world
from agent import AGENT
##import torch
##import torch.backends.cudnn as cudnn

##cudnn.enabled = True # cudnn을 사용하도록 설정. GPU를 사용하고 있으면 기본값은 True 입니다.
##cudnn.benchmark = True # inbuilt cudnn auto-tuner가 사용 중인 hardware에 가장 적합한 알고리즘을 선택하도록 허용합니다.

##device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda가 사용 가능하면 device에 "cuda"를 저장하고 사용 가능하지 않으면 "cpu"를 저장한다.
##torch.cuda.device_count() # 현재 PC의 사용가능한 GPU 사용 갯수 확인


WORLD_HEIGHT = 5
WORLD_WIDTH = 10

env = grid_world(WORLD_HEIGHT,WORLD_WIDTH,
                 GOAL = [[WORLD_HEIGHT-1, WORLD_WIDTH-1]],
                 OBSTACLES=[[0,2], [1,2], [2,2], [2,4], [3,4], [2, 6],[3, 6],[4, 6]])
agent = AGENT(env,is_upload=False)
agent.Monte_Carlo_Control(epsilon=0.4,decay_period=20000, decay_rate=0.9)
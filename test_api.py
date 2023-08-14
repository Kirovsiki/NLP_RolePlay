from fastapi import FastAPI
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.message import Message
from parlai.core.worlds import DialogPartnerWorld
from parlai.agents.local_human.local_human import LocalHumanAgent
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    text: str

def init_parlai():
    # 创建并设置模型
    params = {
        'model_file': 'model/guguAI20',
        'dict_file': 'data/models/blender/blender_3B/model.dict',
        'dict_tokenizer': 'bpe',
        'dict_lower': True,
        'gpu': 0,
        'log_keep_fields': 'all',
        'outfile': 'log_chartchat_log.txt',
        'beam_size': 20,
        'inference': 'nucleus',
        'top_p': 0.9,
        'min_length': 10,
        'max_length': 100,
        'temperature': 1.36,
    }

    parser = ParlaiParser(add_model_args=True)
    parser.set_params(**params)
    opt = parser.parse_args(print_args=False)

    human_agent = LocalHumanAgent(opt)
    agent = create_agent(opt, requireModelExists=True)
    world = DialogPartnerWorld(opt, [human_agent, agent])
    return agent, world

@app.post("/predict/")
async def get_model_response(item: Item):
    agent, world = init_parlai()
    # 设置和观察初始信息
    agent.observe({'text': item.text, 'episode_done': False})
    world.parley()

    # 返回模型的回应
    model_response = agent.act()
    return {"model_response": model_response['text']}

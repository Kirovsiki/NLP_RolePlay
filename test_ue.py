from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent, Agent
from parlai.core.worlds import DialogPartnerWorld
from parlai.utils.world_logging import WorldLogger
from parlai.core.message import Message
import sys
sys.path.append('A:/UE5/guguAI_test/Intermediate/PythonStub')
import unreal

class UnrealEngineAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'UnrealEngineAgent'
        self.episode_done = False
        self.observation = None

    def observe(self, observation):
        self.observation = observation

    def act(self):
        # Get input from Unreal Engine instead of command line
        input_text = self.observation['text']  # Use the observed text as input
        return {'id': self.id, 'text': input_text, 'episode_done': self.episode_done}


def chat(input_text):
    # set up the parameters
    params = {
        'model_file': 'C:\\Users\\97919\\Desktop\\GUGUAI\\ParlAI-main\\model\\guguAI18',
        'dict_file': 'data/models/blender/blender_90M/model.dict',
        'dict_tokenizer': 'bpe',
        'dict_lower': True,
        'gpu': 0,  # add this line to use GPU
        'log_keep_fields': 'all',
        'outfile': 'log_chartchat_log.txt',

        # New parameters for generating longer text
        'beam_size': 20,  # Increase beam size
        'inference': 'nucleus',  # Use nucleus sampling
        'top_p': 0.9,  # Set top_p for nucleus sampling
        'min_length': 10,  # Set minimum length of the generation
        'max_length': 100,  # Set maximum length of the generation
        'temperature': 1.3,
    }

    # create a parser and add command line arguments
    parser = ParlaiParser(add_model_args=True)
    parser.set_params(**params)
    opt = parser.parse_args(args=[], print_args=False)

    agent = create_agent(opt, requireModelExists=True)  # create agent first

    # send identity reminder to the model
    identity_reminder = Message({'text': 'You are guguAI, a female robot.', 'episode_done': False})
    agent.observe(identity_reminder)

    unreal_engine_agent = UnrealEngineAgent(opt)
    unreal_engine_agent.observe({'text': input_text, 'episode_done': False})

    # create world
    world = DialogPartnerWorld(opt, [unreal_engine_agent, agent])

    # let the model respond
    world.parley()

    # get the model's response
    model_response = world.get_acts()[1]['text']

    return model_response

def test_chat():
    print(get_response("Hello, world!"))

#@unreal.ufunction()
def get_response(input_text):
    response = chat(input_text)
    unreal.log(response)
    return response

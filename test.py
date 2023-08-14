from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import DialogPartnerWorld
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.utils.world_logging import WorldLogger
from parlai.core.message import Message

def main():
    # set up the parameters
    params = {
        'model_file': 'model/guguAI19',
        'dict_file': 'data/models/blender/blender_3B/model.dict',
        'dict_tokenizer': 'bpe',
        'dict_lower': True,
        'gpu': 0,  # add this line to use GPU
        'log_keep_fields': 'all',
        'outfile': 'log_chartchat_log.txt',

        # New parameters for generating longer text
        'beam_size': 20,  # Increase beam size
        'inference': 'nucleus',  # Use nucleus sampling
        'top_p': 0.9,  # Set top_p for nucleus sampling
        'min_length': 20,  # Set minimum length of the generation
        'max_length': 100,  # Set maximum length of the generation
        'temperature': 1.3,
    }


    # create a parser and add command line arguments
    parser = ParlaiParser(add_model_args=True)
    parser.set_params(**params)
    opt = parser.parse_args(print_args=False)

    # create agents
    human_agent = LocalHumanAgent(opt)
    agent = create_agent(opt, requireModelExists=True)

    # create world
    world = DialogPartnerWorld(opt, [human_agent, agent])

    # send initial message to the model
    initial_message = Message({'text': 'You are guguAI, a female robot.', 'episode_done': False})
    agent.observe(initial_message)

    # create logger
    logger = WorldLogger(opt)

    # chat
    try:
        while True:
            # let human agent act first
            if agent.observation is None:
                world.parley()
            world.parley()
            logger.log(world)
            if world.epoch_done():
                print("EPOCH DONE")
                break
    except KeyboardInterrupt:
        print('Chat ended with KeyboardInterrupt.')

    # save chat logs
    logger.reset_world()  # this is needed to flush the last few messages
    logger.write_parlai_format(opt['outfile'])

if __name__ == '__main__':
    main()

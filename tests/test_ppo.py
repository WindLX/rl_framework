import argparse

from labml import experiment

from rl_framework.agent.ac import ACEvalAgent, ACEvalConfig
from rl_framework.agent.ppo import PPOAgent, PPOConfig


def train(
    env_name: str,
    config: PPOConfig,
    envs,
    model,
    optimizer,
    lr_scheduler,
):
    experiment.create(name=env_name)

    with PPOAgent(config, envs, model, optimizer, lr_scheduler) as m:
        with experiment.start():
            m.run_training_loop(0)


def eval(config: ACEvalConfig, env, actor):
    with ACEvalAgent(config, env, actor) as m:
        m.run_eval_loop(10)


def test(
    env_name: str,
    config: PPOConfig,
    eval_config: ACEvalConfig,
    envs,
    env,
    model,
    actor,
    optimizer,
    lr_scheduler,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if args.eval:
        eval(eval_config, env, actor)
    else:
        train(env_name, config, envs, model, optimizer, lr_scheduler)

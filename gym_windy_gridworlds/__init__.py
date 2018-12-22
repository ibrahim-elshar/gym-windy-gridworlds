from gym.envs.registration import register

register(
        id='WindyGridWorld-v0',
        entry_point='gym_windy_gridworlds.envs:WindyGridWorldEnv',
        )
register(
        id='StochWindyGridWorld-v0',
        entry_point='gym_windy_gridworlds.envs:StochWindyGridWorldEnv',
        )
register(
        id='KingWindyGridWorld-v0',
        entry_point='gym_windy_gridworlds.envs:KingWindyGridWorldEnv',
        )		
register(
        id='StochKingWindyGridWorld-v0',
        entry_point='gym_windy_gridworlds.envs:StochKingWindyGridWorldEnv',
        )
		
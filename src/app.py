from ast import literal_eval
from raya.application_base import RayaApplicationBase
from raya.tools.image import show_image, draw_on_image
from raya.skills import RayaSkill, RayaSkillHandler

from skills.approach_to_tags import SkillApproachToTags


class RayaApplication(RayaApplicationBase):

    async def setup(self):
        self.skill_apr2tags:RayaSkillHandler = \
                self.register_skill(SkillApproachToTags)
        
        await self.skill_apr2tags.execute_setup(
                setup_args={
                        'working_cameras': self.cameras,
                        'identifier': self.identifier,
                        'tags_size': self.tags_size,
                    },
            )
        

    async def main(self):
        await self.skill_apr2tags.execute_main(execute_args = {
                    'distance_to_goal': self.target_distance
                },
                callback_feedback=self.cb_feedback)


    async def finish(self):
        finish_result= await self.skill_apr2tags.execute_finish()
        self.log.debug(finish_result)


    async def cb_feedback(self, feedback):
        self.log.debug(feedback)

    def get_arguments(self):
        
        self.tags_size = self.get_argument(
                '-s', '--tag_size',
                type=float,
                help='size of tags to be detected',
                required=True
            )
        self.cameras = self.get_argument(
                '-c', '--cameras-name', 
                type=str, 
                list= True, 
                required=True,
                help='name of cameras to use'
            )   
        self.target_distance = self.get_argument(
                '-d', '--distance-to-target', 
                type=float, 
                required=False,
                default=0.5,
                help='Final target distance'
            )  

        self.identifier = self.get_argument(
                '-i', '--identifier', 
                type= int,
                list= True, 
                required=True,
                default='',
                help='ids of the apriltags to be used'
            )  
        self.save_trajectory = self.get_flag_argument(
                '--save-trajectory',
                help='Enable saving trajectory',
            )
        try:
            self.identifier = literal_eval(self.identifier)
        except:
            pass







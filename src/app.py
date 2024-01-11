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
                        'tags_size': self.tags_size,
                    },
            )


    async def main(self):
        execute_result = await self.skill_apr2tags.execute_main(
                execute_args={
                        'distance_to_goal': self.target_distance,
                        'angular_velocity': self.vel_ang,
                        'linear_velocity': self.vel_lin,
                        'step_size': self.step_size,
                        'max_x_error_allowed': self.max_x_err,
                        'max_y_error_allowed': self.max_y_err,
                        'max_angle_error_allowed': self.max_a_err,
                        'max_allowed_distance':self.max_distance,
                        'identifier': self.identifier,
                        'y_offset': self.y_offset,
                        'correct_if_only_one_tag': True,
                    },
                callback_feedback=self.cb_feedback
            )
        self.log.debug(execute_result)


    async def finish(self):
        await self.skill_apr2tags.execute_finish(
            callback_feedback=self.cb_feedback
        )


    async def cb_feedback(self, feedback):
        self.log.debug(feedback)


    def get_arguments(self):
        
        self.tags_size = self.get_argument('-s', '--tag-size',
                type=float,
                help='size of tags to be detected',
                required=True
            )
        self.cameras = self.get_argument('-c', '--cameras', 
                type=str, 
                list=True, 
                required=True,
                help='name of cameras to use'
            )   
        self.target_distance = self.get_argument('-d', '--distance-to-target', 
                type=float, 
                required=False,
                default=0.5,
                help='Final target distance'
            )  
        self.identifier = self.get_argument('-i', '--identifier', 
                type= int,
                list= True, 
                required=True,
                default='',
                help='ids of the apriltags to be used'
            )  
        self.save_trajectory = self.get_flag_argument('--save-trajectory',
                help='Enable saving trajectory',
            )
        self.step_size = self.get_argument('--step-size',
                type=float,
                help='size of tags to be detected',
                default=0.2,
            )
        self.max_x_err = self.get_argument('--max-x-err',
                type=float,
                help='size of tags to be detected',
                default=0.02,
            )
        self.max_y_err = self.get_argument('--max-y-err',
                type=float,
                help='size of tags to be detected',
                default=0.02,
            )
        self.max_a_err = self.get_argument('--max-a-err',
                type=float,
                help='size of tags to be detected',
                default=0.5,
            )
        self.vel_ang = self.get_argument('--vel-ang',
                type=float,
                help='size of tags to be detected',
                default=60.0,
            )
        self.vel_lin = self.get_argument('--vel-lin',
                type=float,
                help='size of tags to be detected',
                default=0.5,
            )
        self.max_distance = self.get_argument(
            '--max-distance',
            type= float,
            help='maximum distance allowed to start approaching',
            default=2.5)
        self.y_offset = self.get_argument(
            '--y-offset',
            type= float,
            help='Offset in y axis',
            default=0.0)           
        try:
            self.identifier = literal_eval(self.identifier)
        except:
            pass

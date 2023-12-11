# Ra-Ya Skill - Approach to Tags
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Unlimited-Robotics/skill_approach_to_tags/graphs/commit-activity)


## Description

This skill is created in order to make an approach to the tags of the objects that the robot sees, it can use 1 or more tags, depending on the size.

Details about the logic can be found in the [Ra-Ya documentation](https://www.notion.so/Approach-to-apriltags-79d01bf78b124e31a15874fd5850d8a8).

## Requirements

* [Ra-Ya controllers]: MotionController, CVController

## Installation

``` bash
rayasdk skills install approach_to_tags
```

## Usage

This example will approach to the tags with identifier 1 and 2 both have a size of 8 cm, using the camera nav_top, when the robot is at 50 cm from the tags it will stop.
``` python
from raya.application_base import RayaApplicationBase
from raya.skills import RayaSkillHandler

from skills.approach_to_tags import SkillApproachToTags

class RayaApplication(RayaApplicationBase):

    async def setup(self):
        self.skill_apr2tags:RayaSkillHandler = \
                self.register_skill(SkillApproachToTags)
        
        await self.skill_apr2tags.execute_setup(
                setup_args={
                        'working_cameras': ['nav_top'],
                        'tags_size': 0.08,
                    },
            )


    async def main(self):
        execute_result = await self.skill_apr2tags.execute_main(
                execute_args={
                        'identifier': [1, 2],
                        'distance_to_goal': 0.5,
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
```

## Exceptions

| Exception | Value (error_code, error_msg) |
| :-------  | :--- |
| ERROR_INVALID_ANGLE | (1, 'Invalid angle, must be between -180 and 180')
| ERROR_INVALID_PREDICTOR | (2, 'Invalid predictor') |
| ERROR_IDENTIFIER_NOT_DEFINED | (3, 'Identifier must be defined') |
| ERROR_NO_TARGET_FOUND | (4, 'Not target found after {NO_TARGET_TIMEOUT_LONG}') |
| ERROR_TOO_DISALIGNED | (5, Custom message) |
| ERROR_TOO_CLOSE_TO_TARGET | (6, Custom message) |
| ERROR_TOO_FAR_TO_TARGET | (7, Custom message) |
| ERROR_IDENTIFIER_LENGTH_HAS_BEEN_EXCEED | (8, 'Maximum length of identifier is 2') |
| ERROR_MOVING | (9, Custom message) |
| ERROR_DISTANCE_TO_CORRECT_TOO_HIGH | (10, Custom message) |
| ERROR_FINAL_ERROR_Y_NOT_ACCOMPLISHED | (11, Custom message) |
| ERROR_NOT_TAG_MISSING_FOUND | (12, Custom message) |

## Arguments

### Setup

#### Required

| Name              | Type     | Description |
| :--------------- | :------: | :---- |
| working_cameras   | [string] | List of cameras to use. Take into account that more cameras means more models to activate. |
| tags_size         | float    | Size of the tags to use, in meters, all tags have to same the same size. |

#### Default

| Name          | Type | Default value | Description |
| :---------------- | :------: | :------: | :---- |
| fsm_log_transitions | boolean | True | Shows the log of the transitions of the fsm. |
| enable_obstacles | boolean | True | Enables the obstacle detection when the robot moves, in case of obstacle it tries to avoid it, If the obstacles does not moves it will raise and exception |

### Execute

#### Required

| Name          | Type | Description |
| :---------------- | :------: | :---- |
| identifier | [int] | List of identifiers of the tags to use, The order of this list must be the same as the order of the tags. Like this: [tag1, tag2, tag3] means that the tag1 is the one on the left, tag2 is the one in the middle and tag3 is the one on the right. |

#### Default


| Name                              | Type    | Default value | Description                                                                                       |
| :---------------------------     | :----:  | :-----------: | :------------------------------------------------------------------------------------------------  |
| distance_to_goal                  | float   | 0.1           | Distance to the goal, in meters; the robot stops upon reaching this distance to the goal and corrects the angle.         |
| angle_to_goal                     | float   | 0.0           | Angle in degrees; approximation angle to target.                                                  |
| angular_velocity                  | int     | 10            | Angular velocity.                                                                                 |
| linear_velocity                   | float   | 0.1           | Linear velocity.                                                                                  |
| min_correction_distance           | float   | 0.5           | Minimum correction distance.                                                                      |
| max_misalignment                  | float   | 1.0           | Maximum misalignment.                                                                             |
| step_size                         | float   | 0.2           | Step size.                                                                                        |
| tags_to_average                   | int     | 6             | Number of tags to average.                                                                        |
| max_x_error_allowed               | float   | 0.02          | Maximum allowed error in the x-axis.                                                              |
| max_y_error_allowed               | float   | 0.05          | Maximum allowed error in the y-axis.                                                              |
| max_angle_error_allowed           | float   | 5.0           | Maximum allowed error in angle.                                                                   |
| allowed_motion_tries              | int     | 10            | Number of allowed motion tries.                                                                   |
| max_allowed_distance              | float   | 2.5           | Maximum allowed distance from target.                                                             |
| scaling_step_to_checking          | float   | 1.5           | Scaling step to checking.                                                                         |
| max_allowed_rotation              | int     | 40            | Maximum rotation allowed in degrees.                                                              |
| min_allowed_rotation_intersection | float   | 5.0           | Minimum allowed rotation at the intersection in degrees.                                          |
| max_reverse_adjust                | float   | 0.2           | Maximum reverse adjustment to correct final angle in meters.                                      |
| max_allowed_correction_tries      | int     | 3             | Maximum allowed correction tries in final correction.                                             |
| enable_initial_reverse_adjust     | boolean | False         | Enable initial reverse adjustment.                                                                |
| enable_final_reverse_adjust       | boolean | False         | Enable final reverse adjustment.                                                                  |
| enable_step_intersection          | boolean | False         | Enable step intersection.                                                                         |
| correct_if_only_one_tag           | boolean | False         | In case that it detects only one tag it will rotate `max_angle_if_oy_one_tag` to see the other tag. |
| max_angle_if_oy_one_tag           | int     | 30            | Maximum angle if `correct_if_only_one_tag` is set to true.                                        |
| y_offset                          | float   | 0.0           | Y-axis offset.                                                                                    |
 

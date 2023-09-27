NO_TARGET_TIMEOUT_LONG = 3.0
NO_TARGET_TIMEOUT_SHORT= 0.4

HANDLER_NAMES = {
        'TagsDetectorHandler': 'tag_id',
}

MAX_MISALIGNMENT = 1.0
MAXIMUM_VALID_TIME_DETECTION = 0.2
X_THRESHOLD_ERROR = 0.02
Y_THRESHOLD_ERROR = 0.02
ERROR_INVALID_ANGLE = (1, f'Invalid angle, must be between -180 and 180')
ERROR_INVALID_PREDICTOR = (2, f'Invalid predictor')
ERROR_IDENTIFIER_NOT_DEFINED = (3, f'Identifier must be defined')
ERROR_NO_TARGET_FOUND = (4, f'Not target found after {NO_TARGET_TIMEOUT_LONG}')
ERROR_TOO_DISALIGNED = 5
ERROR_TOO_CLOSE_TO_TARGET = 6
ERROR_TOO_FAR_TO_TARGET = 7
ERROR_IDENTIFIER_LENGTH_HAS_BEEN_EXCEED = (8, 
                                           'Maximum length of identifier is 2')
ERROR_MOVING = 9
ERROR_DISTANCE_TO_CORRECT_TOO_HIGH = 10
ERROR_FINAL_ERROR_Y_NOT_ACCOMPLISHED = 11
ERROR_NOT_TAG_MISSING_FOUND = 12
from ultralytics.nn.modules.head import *
from ultralytics.engine.model import Model as UltralyticsModel
from ultralytics.nn.modules import Concat

heads = (Detect, Segment, Pose, Classify, OBB, RTDETRDecoder, v10Detect, YOLOEDetect, YOLOESegment)
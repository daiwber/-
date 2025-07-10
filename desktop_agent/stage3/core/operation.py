import time



# 操作历史记录
class Operation:
    def __init__(self, op_type, target=None, value=None, coordinates=None, description="", vis_image=None):
        self.type = op_type.upper() if isinstance(op_type, str) else str(op_type)
        self.target = target
        self.value = value
        self.coordinates = coordinates
        self.description = description
        self.timestamp = time.time()
        self.success = True
        self.vis_image = vis_image

    def to_dict(self):
        return {
            "type": self.type,
            "target": self.target,
            "value": self.value,
            "coordinates": self.coordinates,
            "description": self.description,
            "timestamp": self.timestamp,
            "success": self.success
        }
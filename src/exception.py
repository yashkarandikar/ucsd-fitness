class InvalidValueException(Exception):
    
    def __init__(self, param, value):
        self.param = param
        self.value = value

    def __str__(self):
        return "Param : %s, value = %s" % (self.param, self.value)

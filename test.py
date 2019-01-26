class Class1(object):
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self):
        raise NotImplementedError()


class Class2(Class1):
    
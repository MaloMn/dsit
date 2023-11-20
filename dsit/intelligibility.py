from dsit.model import Model
from dsit.preprocessing import Data


class Intelligibility:

    def __init__(self, model: Model, data: Data):
        self.model = model
        self.data = data

    def get_score(self) -> float:
        return 10.0


if __name__ == '__main__':
    pass

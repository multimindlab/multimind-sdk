class MinimalAgent:
    """
    A minimal agent for demonstration. It echoes or transforms input data.
    """
    def __init__(self, name: str, transform: str = None):
        self.name = name
        self.transform = transform

    async def run(self, input_data):
        if self.transform == 'upper':
            return str(input_data).upper()
        elif self.transform == 'reverse':
            return str(input_data)[::-1]
        else:
            return f"{self.name} received: {input_data}" 
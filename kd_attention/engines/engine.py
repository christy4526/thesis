class TrainingEngine(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if isinstance(exc_type, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    import time
    with TrainingEngine() as t:
        while True:
            time.sleep(1)

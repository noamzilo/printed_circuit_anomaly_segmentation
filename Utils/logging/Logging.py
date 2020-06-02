class Logging(object):

    @staticmethod
    def log_in_out(func):
        def logging_decorator(*args, **kwargs):
            print("Enter " + func.__name__)
            result = func(*args, **kwargs)
            print("Leave " + func.__name__)
            return result

        return logging_decorator

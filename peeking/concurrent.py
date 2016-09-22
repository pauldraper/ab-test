import contextlib
import functools
import multiprocessing
import sys
import traceback

class InstanceMethod:
    def __init__(self, f, *args, **kwargs):
        self.name = f.im_func.func_name
        self.self = f.im_self
        self.args = args
        self.kwargs = kwargs
    def __call__(self, *args, **kwargs):
        return functools.partial(getattr(self.self, self.name), *self.args, **self.kwargs)(*args, **kwargs)

class Wrapper:
    def __init__(self, f):
        self.f = f
    def __call__(self, arguments):
        try:
            return self.f(*arguments)
        except:
            raise Exception(''.join(traceback.format_exception(*sys.exc_info())))

@contextlib.contextmanager
def run(f, runs, arguments = ((),)):
    if hasattr(f, 'im_func') and hasattr(f, 'im_self'):
        f = InstanceMethod(f)
    pool = multiprocessing.Pool()
    try:
        def values():
            progress = 0
            total = len(arguments) * runs
            sys.stdout.write('{}/{}'.format(progress, total))
            sys.stdout.flush()
            for r in pool.imap_unordered(Wrapper(f), arguments * runs):
                progress += 1
                sys.stdout.write('\r{}/{}'.format(progress, total))
                sys.stdout.flush()
                yield r
        yield values()
    finally:
        print
        pool.close()

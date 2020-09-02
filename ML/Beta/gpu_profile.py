import datetime
import linecache
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'    # disables async kernel launches

from py3nvml import py3nvml
import torch
import gc

print_tensor_sizes = False
use_incremental = False

if 'GPU_DEBUG' in os.environ:
    gpu_profile_fn = f"GPU_{os.environ['GPU_DEBUG']}_mem_prof-{datetime.datetime.now():%d-%b-%y-%H-%M-%S}.prof.txt"
    print("Profiling at location {}".format(gpu_profile_fn))

last_tensor_sizes = set()
last_meminfo_used = 0
lineno = None
func_name = None
filename = None
module_name = None

def gpu_profile(frame, event):
    global last_meminfo_used, last_tensor_sizes
    global lineno, func_name, filename, module_name

    if event == 'line':
        try:
            if lineno:
                py3nvml.nvmlInit()
                handle = py3nvml.nvmlDeviceGetHandleByIndex(int(os.environ["GPU_DEBUG"]))
                meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                line = linecache.getline(filename, lineno)
                where_str = module_name+ ' '+func_name+' '+str(lineno)

                new_meminfo_used = meminfo.used
                mem_display = new_meminfo_used-last_meminfo_used if use_incremental else new_meminfo_used
                with open(gpu_profile_fn, "a+") as f:
                    f.write(f"{where_str:<50}"
                            f":{(mem_display) / 1024 ** 2:<7.1f}Mb "
                            f"{line.rstrip()}\n")

                    last_meminfo_used = new_meminfo_used
                    if print_tensor_sizes:
                        for tensor in get_tensors():
                            if not hasattr(tensor, 'dbg_alloc_where'):
                                tensor.dbg_alloc_where = where_str
                        new_tensor_sizes = {(type(x), tuple(x.size()), x.dbg_alloc_where)
                                            for x in get_tensors()}

                        for t, s, loc in new_tensor_sizes - last_tensor_sizes:
                            f.write(f'+ {loc:<50} {str(s):<20} {str(t):<10}\n')

                        for t, s, loc in last_tensor_sizes - new_tensor_sizes:
                            f.write(f'- {loc:<50} {str(s):<20} {str(t):<10}\n')

                        last_tensor_sizes = new_tensor_sizes
                py3nvml.nvmlShutdown()

            lineno = None

            func_name = frame.f_code.co_name
            filename = frame.f_globals["__file__"]
            module_name = frame.f_globals["__name__"]
            lineno = frame.f_lineno

            if 'Beta' not in os.path.dirname(os.path.abspath(filename)):
                lineno = None

            return gpu_profile

        except (KeyError, AttributeError):
            pass

    return gpu_profile


def get_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj["data"]):
                tensor = obj["data"]
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass

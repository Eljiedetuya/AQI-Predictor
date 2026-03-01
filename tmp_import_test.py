import importlib.util
p='notebooks/model_trainer.py'
spec=importlib.util.spec_from_file_location('mt', p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('Imported notebooks/model_trainer.py OK')

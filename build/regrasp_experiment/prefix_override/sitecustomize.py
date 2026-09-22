import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/heroes/Workspace/eksperimen_R2/grasp_ws/install/regrasp_experiment'

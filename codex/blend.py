import numpy as np
base=np.load('/tmp/P_f3_noip_400.npy');sig=np.load('codex/signature_P.npy');bag=np.load('/tmp/P_bag4_noip.npy')
np.save('codex/base_signature_blend_P.npy',(base+sig)/2)
np.save('codex/bag_signature_blend_P.npy',(bag+sig)/2)

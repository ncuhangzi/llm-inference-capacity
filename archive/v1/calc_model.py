# -*- coding: utf-8 -*-
"""Analytic parameter / cache model for Qwen3.5-122B-A10B and Qwen3.8-27B,
cross-validated against real HuggingFace safetensors byte counts."""
import json, sys
io = sys.stdout

def gdn_params(h, nk, nv, dk, dv, conv_k):
    kdim = nk*dk; vdim = nv*dv; conv_dim = 2*kdim + vdim
    p = {}
    p['in_proj_qkv'] = conv_dim*h          # q,k (kdim each) + v
    p['in_proj_z']   = vdim*h              # output gate
    p['in_proj_a']   = nv*h
    p['in_proj_b']   = nv*h
    p['out_proj']    = h*vdim
    p['conv1d']      = conv_dim*conv_k
    p['A_log']       = nv
    p['dt_bias']     = nv
    p['norm']        = dv
    return p, conv_dim

def attn_params(h, nq, nkv, hd, out_gate=True):
    q_out = nq*hd*(2 if out_gate else 1)
    return {'q_proj': q_out*h, 'k_proj': nkv*hd*h, 'v_proj': nkv*hd*h,
            'o_proj': h*nq*hd, 'q_norm': hd, 'k_norm': hd}

def vision_params(depth=27, h=1152, inter=4304, out_h=3072, patch=16, tp=2, ch=3, npos=2304, merge=2):
    p = {}
    p['patch_embed'] = ch*tp*patch*patch*h + h
    p['pos_embed']   = npos*h
    per = (3*h*h + 3*h) + (h*h + h) + (h*inter + inter) + (inter*h + h) + 4*h
    p['blocks'] = depth*per
    mh = out_h  # merger hidden
    p['merger'] = (h*merge*merge)*mh + mh + mh*out_h + out_h + 2*(h*merge*merge)
    return p

# ---------------- Qwen3.5-122B-A10B ----------------
def q35():
    h=3072; L=48; nfull=12; ngdn=36; V=248320
    nk,nv,dk,dv,ck = 16,64,128,128,4
    E=256; me=1024; se=1024
    g,conv_dim = gdn_params(h,nk,nv,dk,dv,ck)
    a = attn_params(h,32,2,256)
    moe = {'experts': E*3*h*me, 'shared': 3*h*se, 'gate': E*h, 'shared_gate': h}
    out={'name':'Qwen3.5-122B-A10B','h':h,'L':L,'nfull':nfull,'ngdn':ngdn,'conv_dim':conv_dim,
         'gdn_per_layer':sum(g.values()),'attn_per_layer':sum(a.values()),
         'moe_per_layer':sum(moe.values()),'norms_per_layer':2*h,
         'embed':V*h,'lm_head':V*h,'final_norm':h,
         'gdn_break':g,'attn_break':a,'moe_break':moe}
    out['lang'] = (ngdn*out['gdn_per_layer'] + nfull*out['attn_per_layer']
                   + L*out['moe_per_layer'] + L*out['norms_per_layer']
                   + out['embed'] + out['lm_head'] + out['final_norm'])
    out['mtp'] = (out['attn_per_layer'] + out['moe_per_layer'] + out['norms_per_layer']
                  + h*2*h + 3*h)   # fc: [h, 2h] + pre_fc norms + mtp norm
    out['vision'] = sum(vision_params(out_h=h).values())
    out['total'] = out['lang']+out['mtp']+out['vision']
    return out

# ---------------- Qwen3.8-27B ----------------
def q38():
    h=5120; L=64; nfull=16; ngdn=48; V=248320; inter=17408
    nk,nv,dk,dv,ck = 16,48,128,128,4
    g,conv_dim = gdn_params(h,nk,nv,dk,dv,ck)
    a = attn_params(h,24,4,256)
    mlp = {'gate_proj':h*inter,'up_proj':h*inter,'down_proj':inter*h}
    out={'name':'Qwen3.8-27B','h':h,'L':L,'nfull':nfull,'ngdn':ngdn,'conv_dim':conv_dim,
         'gdn_per_layer':sum(g.values()),'attn_per_layer':sum(a.values()),
         'moe_per_layer':sum(mlp.values()),'norms_per_layer':2*h,
         'embed':V*h,'lm_head':V*h,'final_norm':h,
         'gdn_break':g,'attn_break':a,'moe_break':mlp}
    out['lang'] = (ngdn*out['gdn_per_layer'] + nfull*out['attn_per_layer']
                   + L*out['moe_per_layer'] + L*out['norms_per_layer']
                   + out['embed'] + out['lm_head'] + out['final_norm'])
    out['mtp'] = (out['attn_per_layer'] + out['moe_per_layer'] + out['norms_per_layer']
                  + h*2*h + 3*h)
    out['vision'] = sum(vision_params(out_h=h).values())
    out['total'] = out['lang']+out['mtp']+out['vision']
    return out

def q332():
    h=5120;L=64;V=151936;inter=25600
    a={'q_proj':64*128*h,'k_proj':8*128*h,'v_proj':8*128*h,'o_proj':h*64*128,'q_norm':128,'k_norm':128}
    mlp=3*h*inter
    lang=L*(sum(a.values())+mlp+2*h)+2*V*h+h
    return {'name':'Qwen3-32B','total':lang,'attn_per_layer':sum(a.values()),'mlp_per_layer':mlp}

REAL = {
 'Qwen/Qwen3.5-122B-A10B': 250173007840,
 'Qwen/Qwen3.8-27B': 55562855904,
 'Qwen/Qwen3-32B': 65524246528,
 'nvidia/Qwen3.5-122B-A10B-NVFP4': 83474870752,
 'Qwen/Qwen3.8-27B-FP8': 3082220272*2 + 24699207680,
 'incoai/Qwen3.8-27B-DFlash2': 1924404480*2,
}

for f,key in [(q35,'Qwen/Qwen3.5-122B-A10B'),(q38,'Qwen/Qwen3.8-27B'),(q332,'Qwen/Qwen3-32B')]:
    m=f(); real=REAL[key]/2
    print('=== %s' % m['name'])
    print('  analytic params : %,d' % m['total'] if False else '  analytic params : {:,}'.format(m['total']))
    print('  real params(BF16 bytes/2): {:,}'.format(int(real)))
    print('  ratio: %.5f' % (m['total']/real))
    for k in ['lang','mtp','vision','gdn_per_layer','attn_per_layer','moe_per_layer','conv_dim']:
        if k in m: print('    %-16s {:,}'.format(m[k]) % k if False else '    %-16s %s' % (k, format(m[k],',')))

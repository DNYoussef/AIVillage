import torch, random, math, time
from geometry.snapshot import geom_snapshot
from optim.augmented_adam import AugmentedAdam
from training.svf_ops import apply_svf
from training.pid_edgechaos import EdgePID
from meta.geo2z_policy import Geo2Z, Replay

# ── init objects ─────────────────────────────────────────────────────────────
optimizer = AugmentedAdam(model.parameters(), lr=2e-5)
pid       = EdgePID()
geo2z     = Geo2Z()
replay    = Replay(50000)
state     = dict(G=None, pre_grok=False)

def run_level(dataset):
    global state
    for step,(prompt, target, tag) in enumerate(dataset):
        # forward
        logits, H = model(prompt, return_h=True)
        loss_task = torch.nn.functional.cross_entropy(logits, target)
        loss_task.backward()

        # geometry
        state['G'] = geom_snapshot(H.view(-1, H.size(-1)))
        gslow = optimizer._grad_window.abs().mean().item() if optimizer._grad_window is not None else 0.

        # RL reward shaping
        reward = .3*(logits.argmax(-1)==target).float().mean().item() \
               + .5*math.tanh(gslow) \
               + .2*max(0, state['G_prev']['ID_nl']-state['G']['ID_nl']) if step else 0

        # SVF action from meta-policy
        geom_vec = torch.tensor([state['G'][k] for k in ['ID_nl','ID_lin','ratio','entropy']], device=H.device)
        z = geo2z(geom_vec)
        for m in model.modules():
            if isinstance(m, torch.nn.Linear): apply_svf(m, z)

        # Grokfast filter & optimiser
        state['pre_grok'] = gslow > .03 and state['G']['ratio'] < .1
        lr_gain = pid.update(state['G']['ratio'])
        for group in optimizer.param_groups: group['lr'] *= (1+lr_gain)
        optimizer.step(amplify=state['pre_grok']); optimizer.zero_grad()

        # replay
        if reward > .2: replay.add(geom_vec.detach(), z.detach(), reward)

        # grok detection
        if state['pre_grok'] and abs(state['G']['ratio']-.05)<.01:
            huge_reward = 10; break

        state['G_prev'] = state['G']

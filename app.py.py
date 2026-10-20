# app.py - Fresh Cut Fruit Shop: animated, colourful RL dashboard
# Run:  streamlit run app.py     (cafe.py + agent.py in same folder)
import time, random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Fresh Fruit Shop AI", page_icon="🍉", layout="wide")

try:
    from cafe import CafeSim
    from agent import train, policy_from_Q
except Exception as e:
    st.error("Could not import cafe.py / agent.py"); st.exception(e); st.stop()

# ---------------- CSS + animations ----------------
FRUITS = ["🍉", "🍓", "🍊", "🍍", "🥭", "🍇", "🍌", "🍎", "🥝", "🍒"]
floaters = "".join(
    f'<span class="fl" style="left:{(i*10+3)%97}%;animation-duration:{9+i*1.7}s;'
    f'animation-delay:{i*0.9}s;font-size:{22+(i%3)*10}px">{f}</span>'
    for i, f in enumerate(FRUITS))
st.markdown(f"""
<style>
.stApp {{ background: linear-gradient(135deg,#fff7ed 0%,#fdf2f8 45%,#ecfeff 100%); }}
.fl {{ position:fixed; bottom:-60px; opacity:.28; z-index:0; pointer-events:none;
       animation: floatup linear infinite; }}
@keyframes floatup {{ 0%{{transform:translateY(0) rotate(0deg)}}
                      100%{{transform:translateY(-115vh) rotate(360deg)}} }}
.hero {{ background: linear-gradient(270deg,#f97316,#ec4899,#8b5cf6,#06b6d4,#22c55e);
         background-size:400% 400%; animation: grad 12s ease infinite;
         padding:26px 30px; border-radius:22px; color:white; box-shadow:0 10px 30px #0002; }}
.hero h1 {{ margin:0; font-size:2.3rem; }} .hero p {{ margin:4px 0 0; opacity:.95; }}
@keyframes grad {{ 0%{{background-position:0% 50%}} 50%{{background-position:100% 50%}}
                   100%{{background-position:0% 50%}} }}
.kpi {{ border-radius:18px; padding:16px 18px; color:white; box-shadow:0 6px 18px #0002;
        animation: pop .7s cubic-bezier(.2,1.4,.4,1) both; transition:transform .25s; }}
.kpi:hover {{ transform:translateY(-6px) scale(1.04); }}
.kpi .v {{ font-size:1.9rem; font-weight:800; }} .kpi .l {{ font-size:.85rem; opacity:.95; }}
@keyframes pop {{ 0%{{opacity:0;transform:scale(.6) translateY(20px)}}
                  100%{{opacity:1;transform:scale(1) translateY(0)}} }}
.shelf {{ font-size:28px; min-height:44px; }} .shelf span {{ display:inline-block; animation:bounce 1s ease; }}
@keyframes bounce {{ 0%{{transform:translateY(-25px);opacity:0}} 60%{{transform:translateY(4px)}}
                     100%{{transform:translateY(0);opacity:1}} }}
.card {{ background:white; border-radius:18px; padding:16px 20px; box-shadow:0 6px 18px #0001;
         animation: pop .6s both; }}
.bad {{ animation: shake .5s; color:#dc2626; font-weight:700 }}
@keyframes shake {{ 0%,100%{{transform:translateX(0)}} 25%{{transform:translateX(-8px)}} 75%{{transform:translateX(8px)}} }}
</style>
<div>{floaters}</div>
<div class="hero"><h1>🍉 Fresh Cut Fruit Shop — can an AI learn to run it?</h1>
<p>Fresh fruit cups spoil fast. Order too much → waste. Order too little → lost customers. Let the AI learn the sweet spot.</p></div>
""", unsafe_allow_html=True)

def kpi(col, label, value, c1, c2, delay=0):
    col.markdown(f'<div class="kpi" style="background:linear-gradient(135deg,{c1},{c2});'
                 f'animation-delay:{delay}s"><div class="l">{label}</div><div class="v">{value}</div></div>',
                 unsafe_allow_html=True)

# ---------------- helpers ----------------
def eval_policy(env, order_fn, months=300):
    profits = []; demand = sold = spoiled = stockout = orders = 0
    for _ in range(months):
        s = env.reset(); done = False; tot = 0
        while not done:
            r, s, done, info = env.step(order_fn(s)); tot += r
            demand += info['demand']; sold += info['sold']; spoiled += info['spoiled']
            stockout += info['stockout']; orders += info['order']
        profits.append(tot)
    return (np.mean(profits), sold/max(demand, 1), spoiled/max(orders, 1),
            stockout/max(demand, 1))

def simulate_day(p, stock, order, demand):
    order = int(np.clip(order, 0, p['max_stock']))
    available = min(stock+order, p['max_stock'])
    sold = min(available, demand); stockout = max(demand-available, 0); leftover = available-sold
    spoiled = int(round(p['spoil_frac']*leftover)); carry = leftover-spoiled
    profit = sold*p['price']-(order*p['order_cost']+leftover*p['holding_cost']
                              +spoiled*p['spoilage_cost']+stockout*p['stockout_penalty'])
    return round(profit, 2), carry, dict(start=stock, order=order, demand=demand, sold=sold,
            spoiled=spoiled, stockout=stockout, carry=carry, profit=round(profit, 2))

def shelf_html(n, emoji="🍉"):
    return '<div class="shelf">' + "".join(
        f'<span style="animation-delay:{i*0.05}s">{emoji}</span>' for i in range(n)) + '</div>'

# ---------------- sidebar ----------------
st.sidebar.header("🏪 The Business")
spoilage_cost = st.sidebar.slider("Spoilage cost / unit (AED)", 0.0, 5.0, 1.5, 0.1)
stockout_pen = st.sidebar.slider("Stockout penalty / customer (AED)", 0.0, 12.0, 6.0, 0.5)
mean_demand = st.sidebar.slider("Typical daily demand", 3, 12, 7, 1)
spoil_frac = st.sidebar.slider("Fraction that spoils overnight", 0.0, 1.0, 0.25, 0.05)
st.sidebar.header("🧠 The AI")
episodes = st.sidebar.select_slider("Training months (episodes)",
                                    [2000, 5000, 8000, 12000, 15000, 20000], value=12000)
gamma = st.sidebar.slider("γ (values the future)", 0.0, 0.99, 0.95, 0.01)

def make_env():
    return CafeSim(spoilage_cost=spoilage_cost, stockout_penalty=stockout_pen,
                   mean_demand=mean_demand, spoil_frac=spoil_frac)

MAX = make_env().max_stock
params = dict(price=5.0, order_cost=1.0, holding_cost=0.3, spoilage_cost=spoilage_cost,
              stockout_penalty=stockout_pen, spoil_frac=spoil_frac, max_stock=MAX)

if st.sidebar.button("🚀 Train the AI", type="primary"):
    bar = st.progress(0, text="🍓 The AI is practising…")
    env = make_env(); chunks = 10
    Q, curve = train(env, episodes=episodes, gamma=gamma)
    for i in range(chunks):
        bar.progress((i+1)/chunks, text=f"🍊 Learning… {int((i+1)/chunks*100)}%"); time.sleep(0.08)
    bar.empty()
    st.session_state.update(Q=Q, curve=curve, params=params, snapshots=None)
    st.balloons(); st.success("Training complete! Explore the tabs below 👇")

if 'Q' not in st.session_state:
    st.info("👈 Set the business and the AI on the left, then press **Train the AI**.")
    st.stop()

Q = st.session_state['Q']; curve = st.session_state['curve']
policy = policy_from_Q(Q)
ai_order = lambda s: int(policy[int(np.clip(s, 0, MAX))])
env = make_env()

rl_profit, service, spoil_rate, stockout_rate = eval_policy(env, ai_order)
st.write("")
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "💰 Avg monthly profit", f"AED {rl_profit:,.0f}", "#16a34a", "#4ade80", 0)
kpi(k2, "✅ Service level", f"{service*100:.0f}%", "#2563eb", "#38bdf8", .15)
kpi(k3, "🗑️ Spoilage rate", f"{spoil_rate*100:.0f}%", "#f97316", "#fbbf24", .3)
kpi(k4, "🚫 Stockout rate", f"{stockout_rate*100:.0f}%", "#db2777", "#f472b6", .45)
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Learning", "🗺️ Strategy", "🎬 Live Shop", "🏆 Scoreboard", "🎮 You vs AI"])

# ---- Learning (animated replay) ----
with tab1:
    st.subheader("Did it actually learn?")
    w = max(1, len(curve)//100)
    smooth = np.convolve(curve, np.ones(w)/w, mode='valid')
    def draw(upto):
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(smooth[:upto], color="#ec4899", lw=2.5)
        ax.fill_between(range(upto), smooth[:upto], smooth.min(), color="#f9a8d4", alpha=.4)
        ax.set_xlim(0, len(smooth)); ax.set_ylim(smooth.min()-20, smooth.max()+20)
        ax.set_xlabel("Training month"); ax.set_ylabel("Profit (AED)")
        ax.set_title("Monthly profit while learning"); ax.grid(alpha=.2)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        return fig
    slot = st.empty()
    if st.button("▶️ Replay the learning animation"):
        for u in np.linspace(5, len(smooth), 25).astype(int):
            slot.pyplot(draw(u)); plt.close('all'); time.sleep(0.06)
    slot.pyplot(draw(len(smooth)))
    st.caption(f"Started ≈ AED {np.mean(curve[:200]):.0f}/month → ended ≈ AED {np.mean(curve[-200:]):.0f}/month.")

# ---- Strategy ----
with tab2:
    st.subheader("The strategy the AI discovered")
    st.write("For each **stock on hand**, how much does the AI **order**? "
             "Look for: *low stock → order a lot, high stock → order little.*")
    cmap = LinearSegmentedColormap.from_list("f", ["#fef08a", "#fb923c", "#ec4899", "#7c3aed"])
    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.imshow(policy.reshape(1, -1), cmap=cmap, aspect="auto")
    for i, v in enumerate(policy):
        ax.text(i, 0, str(int(v)), ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax.set_yticks([]); ax.set_xticks(range(len(policy))); ax.set_xlabel("Stock on hand →")
    ax.set_title("Learned policy: stock → order amount")
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar(range(len(policy)), policy, color=[cmap(v/max(MAX, 1)) for v in policy])
    ax.set_xlabel("Stock on hand"); ax.set_ylabel("Units to order"); ax.grid(alpha=.2, axis='y')
    st.pyplot(fig)

# ---- Live Shop (animated month playback) ----
with tab3:
    st.subheader("🎬 Watch the AI run your shop, day by day")
    speed = st.slider("Seconds per day", 0.1, 1.5, 0.5, 0.1)
    if st.button("▶️ Open the shop for 30 days"):
        e = make_env(); s = e.reset(); done = False; day = 1; total = 0.0
        hist = []; box = st.empty(); chart = st.empty()
        while not done:
            start = s; o = ai_order(start); r, s, done, info = e.step(o); total += r; hist.append(total)
            bad = f'<span class="bad">😟 {info["stockout"]} customers left empty-handed!</span>' if info["stockout"] else "✅ everyone served"
            box.markdown(f"""<div class="card"><b>Day {day}/30</b> &nbsp;|&nbsp; Morning stock: {start} → ordered <b>{o}</b><br>
            {shelf_html(min(start+o, MAX))}
            Demand: <b>{info['demand']}</b> · Sold: <b>{info['sold']}</b> · Spoiled: {info['spoiled']} 🗑️ · {bad}<br>
            Day profit: <b>AED {info['profit']}</b> · Running total: <b style="color:#16a34a">AED {total:,.0f}</b></div>""",
                         unsafe_allow_html=True)
            chart.line_chart(hist, height=180, color="#ec4899")
            day += 1; time.sleep(speed)
        st.success(f"Month finished! Total profit AED {total:,.0f}")

# ---- Scoreboard ----
with tab4:
    st.subheader("Who runs the better shop?")
    cautious = eval_policy(env, lambda s: 4)[0]
    generous = eval_policy(env, lambda s: 10)[0]
    rnd = eval_policy(env, lambda s: random.randint(0, MAX))[0]
    names = ["Cautious (order 4)", "Random", "Generous (order 10)", "🧠 RL Agent"]
    vals = [cautious, rnd, generous, rl_profit]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.barh(names, vals, color=["#94a3b8", "#38bdf8", "#fb923c", "#22c55e"])
    for i, v in enumerate(vals): ax.text(v, i, f" {v:,.0f}", va="center", fontweight="bold")
    ax.set_xlabel("Average monthly profit (AED)"); ax.grid(alpha=.2, axis='x')
    st.pyplot(fig)
    best_naive = max(cautious, rnd, generous)
    st.metric("AI earns more than the best guessing manager by", f"AED {rl_profit-best_naive:,.0f}/month")

# ---- You vs AI ----
with tab5:
    st.subheader("🎮 You play manager for 10 days — then the AI plays the same days")
    if 'game_demand' not in st.session_state or st.button("🔄 New 10-day challenge"):
        g = np.random.default_rng(); seq = []
        for _ in range(10):
            d = g.poisson(mean_demand)
            if g.random() < 0.2: d += g.poisson(5)
            seq.append(int(d))
        st.session_state.update(game_demand=seq, you_day=0, you_stock=5, you_profit=0.0, you_log=[])
    seq = st.session_state['game_demand']; day = st.session_state['you_day']
    if day < 10:
        st.write(f"**Day {day+1} of 10** — stock this morning: **{st.session_state['you_stock']}**")
        st.markdown(shelf_html(st.session_state['you_stock']), unsafe_allow_html=True)
        order = st.number_input("How many fruit cups do you order?", 0, MAX, 7, key=f"ord{day}")
        if st.button("Open the shop ▶️", key=f"go{day}"):
            p, carry, info = simulate_day(params, st.session_state['you_stock'], order, seq[day])
            st.session_state['you_profit'] += p; st.session_state['you_stock'] = carry
            st.session_state['you_log'].append(info); st.session_state['you_day'] += 1
            st.rerun()
        if st.session_state['you_log']: st.table(st.session_state['you_log'])
    else:
        ai_stock = 5; ai_profit = 0.0
        for d in seq:
            o = ai_order(ai_stock); p, ai_stock, _ = simulate_day(params, ai_stock, o, d); ai_profit += p
        you = st.session_state['you_profit']
        a, b = st.columns(2)
        kpi(a, "🧍 Your total profit", f"AED {you:,.0f}", "#7c3aed", "#a78bfa")
        kpi(b, "🧠 AI's total profit", f"AED {ai_profit:,.0f}", "#16a34a", "#4ade80")
        if ai_profit > you: st.success("The AI ran the more profitable shop — it learned it from rewards alone 🤖")
        else: st.balloons(); st.success("You beat the AI! 🎉")
        st.table(st.session_state['you_log'])
        if st.button("Play again"):
            for k in ['game_demand', 'you_day', 'you_stock', 'you_profit', 'you_log']: st.session_state.pop(k, None)
            st.rerun()

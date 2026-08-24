import jax
import jax.numpy as jnp
import optax
import plotly
import plotly.graph_objects as go
import streamlit as st
from flax import linen as nn

import bprg.simulation as sim
from bprg.preset import nn_config_dict, preset_dict
from bprg.solver import FixpointSolver, NeuralNetwork

st.set_page_config(
    page_title="Bootstrap Percolation in Random Graphs",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.header("Estimate Final Infection")


# Plot config
color_κ = plotly.colors.sequential.YlGnBu
color_μ = color_κ[5]
color_η = [color_κ[2], color_κ[0], color_κ[5]]
color_f = color_κ[3]
color_Ψf = color_κ[5]
color_fmc = plotly.colors.qualitative.Set2[3]
margin = {"l": 5, "r": 5, "t": 30, "b": 5}
legend = {"x": 0.5, "y": 1.1, "orientation": "h", "xanchor": "center"}


# State management
def init_session_state():
    # Initialize session state for tracking preset and config changes
    if "preset" not in st.session_state:
        st.session_state.preset = None
    if "nn_data" not in st.session_state:
        st.session_state.nn_data = None
    if "mc_data" not in st.session_state:
        st.session_state.mc_data = None
    if "nn_config_key" not in st.session_state:
        st.session_state.nn_config_key = None
    if "mc_config_key" not in st.session_state:
        st.session_state.mc_config_key = None


def reset_session_state():
    # Reset everything if preset changed (check AFTER preset_name is defined)
    if st.session_state.preset != preset_name:
        st.session_state.preset = preset_name
        st.session_state.nn_data = None
        st.session_state.mc_data = None
        st.session_state.nn_config_key = None
        st.session_state.mc_config_key = None


init_session_state()

with st.expander("Model configuration", expanded=True):
    # Load preset
    preset_name = st.radio("Select a preset", preset_dict.keys(), index=0, horizontal=True)
    reset_session_state()

    preset = preset_dict[preset_name]
    nn_config = nn_config_dict[preset_name]
    κ = preset.κ
    μ = preset.μ
    η = preset.η
    S = preset.S

    nv_fig = 1000
    nv_mc = 1000
    ns_mc = 100

    s = jnp.linspace(S[0], S[1], nv_fig)
    Δμ_s = μ(s)
    μ_s = Δμ_s * nv_fig

    # Kernel visualization
    fig_κ = go.Figure()
    nx = 50
    x = jnp.linspace(S[0], S[1], nx)
    xx, yy = jnp.meshgrid(x, x, indexing="xy")
    κκ = κ(yy, xx)
    # fig_κ.add_trace(go.Surface(x=yy, y=xx, z=κκ, colorscale=color_κ, showscale=False))

    # Temporal fix for plotly 3d surface triangularization issue
    x_ridge = jnp.linspace(S[0], S[1], 100)
    κ_ridge = κ(x_ridge, x_ridge)
    x_mesh = jnp.concatenate([xx.flatten(), x_ridge])
    y_mesh = jnp.concatenate([yy.flatten(), x_ridge])
    κ_mesh = jnp.concatenate([κκ.flatten(), κ_ridge])
    fig_κ.add_trace(
        go.Mesh3d(
            x=x_mesh, y=y_mesh, z=κ_mesh, colorscale=color_κ, intensity=κ_mesh, intensitymode="vertex", showscale=False
        )
    )

    fig_κ.update_layout(
        scene={"xaxis_title": "y", "yaxis_title": "x", "zaxis_title": "κ(x, y)"},
        height=600,
        # scene_camera={"eye": {"x": -2.25, "y": -0.750, "z": 0.25}},
        scene_camera={"eye": {"x": -0.75, "y": -2.5, "z": 0.15}},
        margin=margin,
    )

    # Measure visualization
    fig_μ = go.Figure()
    Δμ_x = μ(x)
    μ_x = (Δμ_x * nx).round(1)
    fig_μ.add_trace(go.Scatter(x=x[1:-1], y=μ_x[1:-1], mode="lines", fill="tozeroy", line={"color": color_μ}))
    fig_μ.update_layout(
        xaxis_title="s",
        yaxis_title="dμ(s)",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 2]},
        height=150,
        margin={"l": 5, "r": 90, "t": 30, "b": 5},
    )

    # Threshold visualization
    fig_η = go.Figure()
    η_xk = η(x)

    def create_bar(s_val, k_val, η_val, s_width=0.75, k_width=1.0, color="blue"):
        """Create a 3D bar using Mesh3d"""
        x_bar, y_bar, z_bar = jnp.meshgrid(
            jnp.linspace(s_val - s_width / 2, s_val + s_width / 2, 2),
            jnp.linspace(k_val - k_width / 2, k_val + k_width / 2, 2),
            jnp.linspace(0, η_val, 2),
        )
        return go.Mesh3d(
            x=x_bar.flatten(),
            y=y_bar.flatten(),
            z=z_bar.flatten(),
            alphahull=0,
            flatshading=True,
            colorscale=[[0, color], [1, color]],
            intensity=jnp.ones(1),
            showscale=False,
        )

    for i, x_val in enumerate(x):
        for k, color in zip(range(η_xk.shape[1]), color_η):
            fig_η.add_trace(create_bar(x_val, k, η_xk[i, k], color=color))

    fig_η.update_layout(
        scene={
            "xaxis_title": "s",
            "yaxis_title": "k",
            "zaxis_title": "η<sub>k</sub>(s)",
            "yaxis": {"tickmode": "linear", "tick0": 0, "dtick": 1},
            "aspectmode": "data",
            "zaxis": {"range": [0, 1]},
        },
        height=300,
        scene_camera={"eye": {"x": 2.0, "y": -1.5, "z": 0.5}},
        margin=margin,
    )

    # Figure layout
    col_preset = st.columns(2)

    def get_first_docmath(doc: str) -> str:
        """
        Return the docstring up to the end of the first math block
        (either $...$ or $$...$$). Safe for Streamlit rendering.
        """
        # Find the first dollar sign
        first = doc.find("$")
        if first == -1:
            return doc

        # Find the second dollar sign
        second = doc.find("$", first + 1)
        if second == -1:
            return doc  # Malformed, return whole string

        # Check if it's a display math block ($$)
        if second == first + 1:
            # It's display math: $$ ... $$
            third = doc.find("$", second + 1)
            if third == -1:
                return doc
            fourth = doc.find("$", third + 1)
            if fourth == -1:
                return doc
            # Return up to and including the closing $$
            return doc[: fourth + 1]
        else:
            # It's inline math: $ ... $
            # Return up to and including the closing $
            return doc[: second + 1]

    with col_preset[0]:
        st.markdown(get_first_docmath(κ.__doc__))
        st.plotly_chart(fig_κ, width="stretch", key="fig_κ")

    with col_preset[1]:
        st.markdown(get_first_docmath(μ.__doc__))
        st.plotly_chart(fig_μ, width="stretch", key="fig_μ")
        st.markdown(get_first_docmath(η.__doc__))
        st.plotly_chart(fig_η, width="stretch", key="fig_η")


with st.expander("Estimate final fraction of infected vertices", expanded=True):
    st.markdown(r"""
        The least fixed point $\widehat{f}$ of the operator $Ψ_κ$,
        $$
        Ψ_κ [\widehat{f}](s) = \widehat{f}(s),
        $$
        characterizes the asymptotic final fraction of infected vertices:
        $$
        \frac{n_{\text{infected}}}{n} \stackrel{n→∞}{⟶} ∫_{\mathcal{S}} \widehat{f} \mathrm{d} μ(s).
        $$
    """)
    col_nn_config = st.columns(2, vertical_alignment="top")
    with col_nn_config[0], st.expander("Neural fixpoint solver", expanded=True):
        st.write(r"Approximate the fixpoint with neural network $f_{NN} ≈ \widehat{f}$")
        col_nn = st.columns(3)
        with col_nn[0]:
            nn_seed = st.number_input("NN random seed", min_value=0, value=2022, step=1)
        with col_nn[1]:
            nn_layers = st.number_input("# Hidden layers", min_value=1, max_value=5, value=nn_config["layers"], step=1)
        with col_nn[2]:
            nn_nodes = st.number_input("# Nodes per layer", min_value=1, max_value=64, value=nn_config["nodes"], step=1)

        col_nn_run = st.columns([2, 1], vertical_alignment="bottom")
        with col_nn_run[0]:
            nn_epochs = st.slider("# Epochs", min_value=100, max_value=2000, value=nn_config["epochs"], step=100)
        with col_nn_run[1]:
            nn_run = st.button("Run NN", type="primary")

        st_nn_progress = st.empty()

        # Track NN config changes and reset trained data if config changed
        nn_config_key = (nn_seed, nn_layers, nn_nodes, nn_epochs)
        if st.session_state.nn_config_key != nn_config_key:
            st.session_state.nn_config_key = nn_config_key
            st.session_state.nn_data = None  # Reset trained NN results

        class NeuralNetwork(nn.Module):
            @nn.compact
            def __call__(self, x, training: bool = True):
                x = x.reshape(-1, 1)
                x = nn.Dense(nn_nodes)(x)
                x = nn.tanh(x)
                for _ in range(nn_layers - 1):
                    x = nn.Dense(nn_nodes)(x)
                    x = nn.tanh(x)
                x = nn.Dense(1)(x)
                x = nn.sigmoid(x)
                return x.squeeze(-1)

        solver = FixpointSolver(NeuralNetwork(), κ=κ, μ=μ, η=η, S=S)
        key = jax.random.PRNGKey(nn_seed)
        params = solver.fnn.init(key, s)
        f_s = solver.f(params, s)
        Ψf_s = solver.Ψf(params, s)
        int_f = (f_s * Δμ_s).sum()

        lr_schedule = optax.cosine_decay_schedule(init_value=0.05, decay_steps=nn_epochs)
        optimizer = optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-10)
        opt_state = optimizer.init(params)

        fig_f = go.Figure()
        scatter_f = go.Scatter(x=s, y=f_s, mode="lines", name="f<sub>NN</sub>(s)", line={"color": color_f, "width": 3})
        scatter_Ψf = go.Scatter(
            x=s, y=Ψf_s, mode="lines", name="Ψ[f<sub>NN</sub>](s)", line={"dash": "dash", "color": color_Ψf, "width": 3}
        )
        fig_f.add_trace(scatter_f)
        fig_f.add_trace(scatter_Ψf)
        fig_f.update_layout(xaxis={"range": S}, legend=legend, margin=margin, height=400)

    with col_nn_config[1], st.expander("Monte Carlo simulation", expanded=True):
        st.write(r"Estimate final infected fraction via Monte Carlo $f_{MC} ≈ \widehat{f}$")
        col_mc_config = st.columns(4, vertical_alignment="bottom")
        with col_mc_config[0]:
            mc_seed = st.number_input("MC random seed", min_value=0, value=2022, step=1)
        with col_mc_config[1]:
            nv_mc = st.number_input("# Vertices", min_value=500, max_value=10000, value=nv_mc, step=100)
        with col_mc_config[2]:
            ns_mc = st.number_input("# Vertex types", min_value=10, max_value=2000, value=ns_mc, step=10)
        with col_mc_config[3]:
            can_gridtype = nv_mc % ns_mc == 0
            mc_gridtype = st.checkbox(
                "Reduce variance",
                value=can_gridtype,
                disabled=not can_gridtype,
                help="Requires # Vertices divisible by # Types",
            )

        col_mc_run = st.columns([2, 1], vertical_alignment="bottom")
        with col_mc_run[0]:
            mc_nm = st.slider("# Simulations", min_value=100, max_value=2000, value=500, step=100)
        with col_mc_run[1]:
            mc_run = st.button("Run MC", type="primary")

        st_mc_progress = st.empty()

        # Track MC config changes and reset simulated data if config changed
        mc_config_key = (mc_seed, nv_mc, ns_mc, mc_nm)
        if st.session_state.mc_config_key != mc_config_key:
            st.session_state.mc_config_key = mc_config_key
            st.session_state.mc_data = None  # Reset simulated MC results

        mc_s = jnp.linspace(S[0], S[1], ns_mc)
        mc_Δμ_s = μ(mc_s)
        vsid_grid, nv_s_grid = sim._apportion_typeindex(mc_Δμ_s, ns_mc, nv_mc)
        vs_grid = mc_s[vsid_grid]
        η_sk_grid = η(vs_grid)
        denom_grid = nv_s_grid
        denom_sample = nv_mc * mc_Δμ_s
        mc_keys = jax.random.split(jax.random.key(mc_seed), mc_nm)
        fmc_s = jnp.zeros_like(mc_s) * jnp.nan
        int_fmc = fmc_s.sum()
        scatter_g = go.Scatter(
            x=mc_s, y=fmc_s, mode="markers", name="f<sub>MC</sub>(s)", line={"color": color_fmc}, opacity=0.7
        )
        fig_f.add_trace(scatter_g)

    plot_per_iter = st.slider("Update plot per _ iterations", min_value=1, max_value=100, value=50, step=1)

    # Display f plot with stored data or initial state
    def string_fval_nn(val):
        return f"$∫_{{\\mathcal{{S}}}} f_{{NN}}(s) \\mathrm{{d}} μ(s) = {val * 100:.2f}\\%$"

    def string_fval_mc(val):
        return f"$∫_{{\\mathcal{{S}}}} f_{{MC}}(s) \\mathrm{{d}} μ(s) = {val * 100:.2f}\\%$"

    def string_fval(nn_val, mc_val):
        return (
            "Estimated final infected fraction: &nbsp;&nbsp;&nbsp;&nbsp;"
            + string_fval_nn(nn_val)
            + ",&nbsp;&nbsp;&nbsp;&nbsp;"
            + string_fval_mc(mc_val)
        )

    # Result display
    st_val_f = st.empty()
    st_fig_f = st.empty()

    if st.session_state.nn_data is not None:
        f_s, Ψf_s, int_f = st.session_state.nn_data
        with fig_f.batch_update():
            fig_f.data[0].y = f_s
            fig_f.data[1].y = Ψf_s

    if st.session_state.mc_data is not None:
        fmc_s, int_fmc = st.session_state.mc_data
        with fig_f.batch_update():
            fig_f.data[2].y = fmc_s

    st_val_f.markdown(string_fval(int_f, int_fmc))
    st_fig_f.plotly_chart(fig_f, width="stretch", key="fig_f")

    @st.fragment
    def run_nn(params=params, opt_state=opt_state, optimizer=optimizer):
        if nn_run:
            for epoch in range(nn_epochs):
                params, opt_state, _ = solver.train_step(params, opt_state, optimizer)
                if epoch % plot_per_iter == 0 or epoch == nn_epochs - 1:
                    f_s = solver.f(params, s)
                    Ψf_s = solver.Ψf(params, s)
                    int_f = (f_s * Δμ_s).sum()
                    # Store in session state
                    st.session_state.nn_data = (f_s, Ψf_s, int_f)
                    with fig_f.batch_update():
                        fig_f.data[0].y = f_s
                        fig_f.data[1].y = Ψf_s
                    st_fig_f.plotly_chart(fig_f, width="stretch")
                    st_val_f.markdown(string_fval(int_f, int_fmc))
                    st_nn_progress.progress(
                        min((epoch + 1) / nn_epochs, 1.0),
                        text=f"Training neural network... Epoch {epoch}/{nn_epochs}",
                    )
            st_nn_progress.empty()

    @st.fragment
    def run_mc():
        if mc_run:
            ni_s_cum = jnp.zeros_like(mc_s)
            for m, mc_key in enumerate(mc_keys):
                if mc_gridtype:
                    vsid = vsid_grid
                    vk = sim._sample_threshold(η_sk_grid, mc_key)
                    A = sim._sample_adjacency_matrix(vs_grid, κ, nv_mc, mc_key)
                    denom = denom_grid
                else:
                    vsid, _, vk, A = sim._simulate_graph(κ, η, mc_s, mc_Δμ_s, nv_mc, mc_key)
                    denom = denom_sample
                vk_final = sim._simulate_percolation_final(vk, A)
                ni_s = sim.count_infected_by_typeindex(vsid, vk_final, ns_mc)
                ni_s_cum += ni_s
                if m % plot_per_iter == 0 or m == mc_nm - 1:
                    fmc_s = ni_s_cum / (m + 1) / denom
                    int_fmc = (fmc_s * mc_Δμ_s).sum()
                    # Store in session state
                    st.session_state.mc_data = (fmc_s, int_fmc)
                    with fig_f.batch_update():
                        fig_f.data[2].y = fmc_s
                    st_fig_f.plotly_chart(fig_f, width="stretch")
                    st_val_f.markdown(string_fval(int_f, int_fmc))
                    st_mc_progress.progress(
                        min((m + 1) / mc_nm, 1.0),
                        text=f"Running Monte Carlo simulations... {m}/{mc_nm}",
                    )
            st_mc_progress.empty()

    # Train neural network if button clicked
    if nn_run:
        run_nn()

    # Run Monte Carlo simulation if button clicked
    if mc_run:
        run_mc()

import json

import jax
import jax.numpy as jnp
import streamlit as st

from bprg.preset import preset_dict
from bprg.simulation import simulate_graph, simulate_percolation_history

st.set_page_config(
    page_title="BPRG:Bootstrap Percolation in Random Graphs",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("# BPRG: Bootstrap Percolation in Random Graphs")

# Configuration
bgcolor = "rgb(15,17,23)" if st.context.theme.type == "dark" else "rgb(255,255,255)"
color_map = {0: "#FD3216", 1: "#FECB52", 2: "#3283FE"}
size_map = {0: 15, 1: 15, 2: 20}
symbol_map = {0: "diamond", 1: "circle", 2: "circle"}
height = 600

st_plot = st.empty()


@st.fragment
def plot_graph():
    # ---------------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------------
    with st.columns(3)[-1], st.expander("Change graph", expanded=False):
        col_param = st.columns(2)
        with col_param[0]:
            preset_name = st.selectbox("Kernel", preset_dict.keys(), index=0, help="See the next page for details.")
            preset = preset_dict[preset_name]
            κ = preset.κ
            μ = preset.μ
            η = preset.η
            S = preset.S

        with col_param[1]:
            seed = st.number_input("Seed", min_value=0, value=2022, step=1, help="Random seed for reproducibility.")
            key = jax.random.key(seed)
            camera_theta = jax.random.uniform(key, (), minval=0.0, maxval=2 * jnp.pi)
        with col_param[0]:
            nv = st.slider("# Vertices", min_value=50, max_value=500, value=100, step=1)
        with col_param[1]:
            ns = st.slider("# Types", min_value=10, max_value=100, value=50, step=1)

    # ---------------------------------------------------------------
    # Simulate Random Graph
    # ---------------------------------------------------------------
    _, _, vk, A = simulate_graph(κ, μ, η, S, nv, ns, key=key)
    e_all = jnp.array(jnp.argwhere(A == 1))
    vk_hist, ea_hist = simulate_percolation_history(vk, A)
    color_hist = [[color_map[int(k)] for k in vk] for vk in vk_hist]
    size_hist = [[size_map[int(k)] for k in vk] for vk in vk_hist]
    symbol_hist = [[symbol_map[int(k)] for k in vk] for vk in vk_hist]

    key_dir, key_r = jax.random.split(key)
    direction = jax.random.normal(key_dir, (nv, 3))
    direction /= jnp.linalg.norm(direction, axis=1, keepdims=True)
    radius = jax.random.uniform(key_r, (nv, 1)) ** (1.0 / 3.0)
    vpos = radius * direction

    # ---------------------------------------------------------------
    # Convert data to JSON for browser
    # ---------------------------------------------------------------
    vertex_data = {
        "x": vpos[:, 0].tolist(),
        "y": vpos[:, 1].tolist(),
        "z": vpos[:, 2].tolist(),
    }

    e_x, e_y, e_z = [], [], []
    for e in e_all:
        p1 = vpos[e[0]]
        p2 = vpos[e[1]]
        e_x.extend([float(p1[0]), float(p2[0]), None])
        e_y.extend([float(p1[1]), float(p2[1]), None])
        e_z.extend([float(p1[2]), float(p2[2]), None])

    ea_x_hist, ea_y_hist, ea_z_hist = [], [], []
    for mask in ea_hist:
        src, dst = jnp.nonzero(mask)
        ea_x, ea_y, ea_z = [], [], []
        for s, d in zip(src.tolist(), dst.tolist()):
            p1, p2 = vpos[s], vpos[d]
            ea_x.extend([float(p1[0]), float(p2[0]), None])
            ea_y.extend([float(p1[1]), float(p2[1]), None])
            ea_z.extend([float(p1[2]), float(p2[2]), None])
        ea_x_hist.append(ea_x)
        ea_y_hist.append(ea_y)
        ea_z_hist.append(ea_z)

    edge_data = {"x": e_x, "y": e_y, "z": e_z}
    active_edge_data = {"x": ea_x_hist, "y": ea_y_hist, "z": ea_z_hist}
    animation_data = {"colors": color_hist, "symbols": symbol_hist, "sizes": size_hist}

    vertex_json = json.dumps(vertex_data)
    edge_json = json.dumps(edge_data)
    active_edge_json = json.dumps(active_edge_data)
    animation_json = json.dumps(animation_data)

    # ---------------------------------------------------------------
    # Browser-side Plotly animation
    # ---------------------------------------------------------------
    html = f"""
    <div id="plot" style="width:100%;height:{height};"></div>

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <script>
    const vertices = {vertex_json};
    const edges = {edge_json};
    const activeEdges = {active_edge_json};
    const animation = {animation_json};

    const nFrames = 360;
    const graph_iter_speed = 2.0;
    const camera_radius = 0.85;
    const camera_height = 0.25;
    const camera_speed = 0.15;
    let camera_theta = {camera_theta};

    const vertexTrace = {{
        x: vertices.x,
        y: vertices.y,
        z: vertices.z,
        mode: "markers",
        type: "scatter3d",
        marker: {{
            size: animation.sizes[0],
            color: animation.colors[0],
            symbol: animation.symbols[0],
            opacity: 1.0,
        }},
        hoverinfo: "none",
        showlegend: false,
    }};

    const edgeTrace = {{
        x: edges.x,
        y: edges.y,
        z: edges.z,
        mode: "lines",
        type: "scatter3d",
        line: {{ color: "grey", width: 2 }},
        opacity: 0.2,
        hoverinfo: "none",
        showlegend: false,
    }};

    const activeEdgeTrace = {{
        x: activeEdges.x[0],
        y: activeEdges.y[0],
        z: activeEdges.z[0],
        mode: "lines",
        type: "scatter3d",
        line: {{ color: {color_map}[0], width: 3 }},
        opacity: 0.2,
        hoverinfo: "none",
        showlegend: false,
    }};

    const layout = {{
        height:{height},
        margin:{{ l:0, r:0, b:0, t:0 }},
        scene:{{
            xaxis:{{ visible:false }},
            yaxis:{{ visible:false }},
            zaxis:{{ visible:false }},
            bgcolor: "{bgcolor}",
            aspectmode:"data",
            camera:{{
                projection:{{ type:"perspective" }}
            }}
        }},
        scene_camera:{{
            eye:{{ x:camera_radius * Math.cos(camera_theta),
            y:camera_radius * Math.sin(camera_theta), z:camera_height + 0.25 * Math.sin(-2.0 * camera_theta) }}
        }}
    }};

    Plotly.newPlot(
        "plot",
        [
            vertexTrace,
            edgeTrace,
            activeEdgeTrace
        ],
        layout,
        {{
            displayModeBar:false,
            responsive:true,
        }}
    );

    let frame = 0;
    function animate()
    {{
        frame = (frame + camera_speed) % nFrames;
        camera_theta = {camera_theta} + 2*Math.PI*frame/nFrames;

        Plotly.relayout(
            "plot",
            {{
                "scene.camera.eye.x": camera_radius*Math.cos(camera_theta),
                "scene.camera.eye.y": camera_radius*Math.sin(camera_theta),
                "scene.camera.eye.z": camera_height + 0.25 * Math.sin(-2.0 * camera_theta),
            }}
        );

        let iter = Math.floor(
            ((graph_iter_speed * frame % nFrames) / nFrames) *
            animation.colors.length
        );

        Plotly.restyle(
            "plot",
            {{
                "marker.color":
                    [animation.colors[iter]],

                "marker.symbol":
                    [animation.symbols[iter]],

                "marker.size":
                    [animation.sizes[iter]],
            }},
            [0]
        );
        requestAnimationFrame(
            animate
        );

        Plotly.restyle(
            "plot",
            {{
                x: [activeEdges.x[iter]],
                y: [activeEdges.y[iter]],
                z: [activeEdges.z[iter]],
            }},
            [2]
        );

    }}

    animate();
    </script>
    """

    st_plot.iframe(html, width="stretch")


plot_graph()

st.markdown("""
`BPRG` is a high-performance Python library for analyzing **bootstrap percolation in random graphs**. It implements the theoretical framework in [Detering and Lin (2026), Bootstrap percolation in random graphs of unbounded rank](https://doi.org/10.1017/apr.2026.10072), and provides a comprehensive set of tools for studying contagion dynamics in random graphs.
""")


st.markdown(r"""
## What is bootstrap percolation?

Bootstrap percolation describes how an **infection, activation, or cascade spreads through a network**. Each vertex is assigned a threshold that specifies how many of its neighbors must be infected before it becomes infected itself. The process starts with a set of initially infected vertices. As soon as an uninfected vertex has at least as many infected neighbors as its threshold, it also becomes infected. This may trigger further infections, and the process continues until no additional vertices can become infected. The main quantity of interest is the **final fraction of infected vertices** and how it depends on the structure of the network, the vertex types, and their thresholds.

## Explore the theory and finite-network behavior

This interactive webpage illustrates the main results of our work on bootstrap percolation in inhomogeneous random graphs. In the model, each vertex has a **type** and a **threshold**, and connections between vertices of types $x$ and $y$ are determined by a user-specified kernel $\kappa(x,y)$. Starting from an initially infected set, a vertex becomes infected once sufficiently many of its neighbors are infected. Our theoretical results determine, for large networks, the final fraction of infected vertices through the least fixed point $\hat f$ of a nonlinear operator,

$$
\hat f = \Psi_\kappa[\hat f],
$$

where $\hat f(x)$ describes the asymptotic infection level for vertices of type $x$. On this webpage, you can specify the kernel and use our **neural-network-based solver** to approximate this fixed point and obtain the theoretical large-network prediction. You can then compare this prediction with the outcome for a **finite network of size $n$**. For this finite-$n$ calculation, vertex types and thresholds are first sampled from their prescribed distribution; the resulting bootstrap percolation process is then evaluated using the efficient matrix-based implementation developed for the numerical analysis. In this way, the webpage provides a direct visual comparison between the **asymptotic theory** and the behavior of finite random networks, and allows you to explore how this comparison changes with the kernel, the network size, and the model parameters.

The webpage also serves as an interactive illustration of **`BPRG`**, our high-performance Python library implementing these methods for analyzing bootstrap percolation in random graphs.

## References

`BPRG` implements the theoretical framework in [Detering and Lin (2026), *Bootstrap Percolation in Random Graphs of Unbounded Rank*](https://doi.org/10.1017/apr.2026.10072).


```
@article{detering2026bootstrap,
  title={Bootstrap Percolation in Random Graphs of Unbounded Rank},
  author={Detering, Nils and Lin, Jimin},
  journal={Advances in Applied Probability},
  year={2026},
  doi={doi.org/10.1017/apr.2026.10072}
}
```
""")

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots
from skimage import data


gato = data.chelsea()

fig = make_subplots(1, 2, subplot_titles=('hola1', 'hola2'))
img = data.camera()/255.0
# fig.add_trace(go.Heatmap(z=img, colorscale='RdBu_r'), 1, 1)
fig.add_trace(go.Image(z=gato), 1, 1)
fig.add_trace(go.Heatmap(z=img, colorscale='Viridis'), 1, 2)
fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
# fig.update_xaxes(constrain='domain', )
fig.update_layout(showlegend=False)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
#fig.update_traces(showscale=False) # oculta color bar
html_str = py.plot(fig, output_type='div')

f = open('prueba.html', 'w')
f.write(html_str)
f.close()

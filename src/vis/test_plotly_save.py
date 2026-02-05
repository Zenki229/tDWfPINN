import plotly.graph_objects as go 
import plotly.io as pio

fig=go.Figure(data=go.Scatter(x=[1,2,3], y=[1,2,3]))

pio.write_image(fig, './test_save.png')

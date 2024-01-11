# state file generated using paraview version 5.10.0-RC1

# uncomment the following three lines to ensure this script works in future versions
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1147, 344]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.OrientationAxesVisibility = 0
renderView1.CenterOfRotation = [0.014576845802366734, 0.009999999776482582, 0.0012499999720603228]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [0.014576845802366734, 0.009999999776482582, 0.046108581076971515]
renderView1.CameraFocalPoint = [0.014576845802366734, 0.009999999776482582, -0.022361531965451407]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.01772136927570156

# Create a new 'Render View'
renderView10 = CreateView('RenderView')
renderView10.ViewSize = [1328, 198]
renderView10.AxesGrid = 'GridAxes3DActor'
renderView10.CenterOfRotation = [0.014669972471892834, 0.009999999776482582, 0.0012499999720603228]
renderView10.StereoType = 'Crystal Eyes'
renderView10.CameraPosition = [0.014669972471892834, 0.009999999776482582, 0.07001638501476622]
renderView10.CameraFocalPoint = [0.014669972471892834, 0.009999999776482582, 0.0012499999720603228]
renderView10.CameraFocalDisk = 1.0
renderView10.CameraParallelScale = 0.017798050111905404

# Create a new 'Render View'
renderView11 = CreateView('RenderView')
renderView11.ViewSize = [1328, 261]
renderView11.AxesGrid = 'GridAxes3DActor'
renderView11.CenterOfRotation = [0.014662092551589012, 0.009999999776482582, 0.0012499999720603228]
renderView11.StereoType = 'Crystal Eyes'
renderView11.CameraPosition = [0.014662092551589012, 0.009999999776482582, 0.0699912924498667]
renderView11.CameraFocalPoint = [0.014662092551589012, 0.009999999776482582, 0.0012499999720603228]
renderView11.CameraFocalDisk = 1.0
renderView11.CameraParallelScale = 0.017791555678218936

# Create a new 'Render View'
renderView12 = CreateView('RenderView')
renderView12.ViewSize = [1328, 190]
renderView12.AxesGrid = 'GridAxes3DActor'
renderView12.CenterOfRotation = [0.014662092551589012, 0.009999999776482582, 0.0012499999720603228]
renderView12.StereoType = 'Crystal Eyes'
renderView12.CameraPosition = [0.014662092551589012, 0.009999999776482582, 0.0699912924498667]
renderView12.CameraFocalPoint = [0.014662092551589012, 0.009999999776482582, 0.0012499999720603228]
renderView12.CameraFocalDisk = 1.0
renderView12.CameraParallelScale = 0.017791555678218936

# Create a new 'Render View'
renderView13 = CreateView('RenderView')
renderView13.ViewSize = [1328, 261]
renderView13.AxesGrid = 'GridAxes3DActor'
renderView13.CenterOfRotation = [0.015262570232152939, 0.009999999776482582, 0.0012499999720603228]
renderView13.StereoType = 'Crystal Eyes'
renderView13.CameraPosition = [0.015262570232152939, 0.00750221868878081, 0.07187133272562819]
renderView13.CameraFocalPoint = [0.015262570232152939, 0.009999999776482582, 0.0012499999720603228]
renderView13.CameraViewUp = [0.0, 0.9993751155644066, 0.03534654708212704]
renderView13.CameraFocalDisk = 1.0
renderView13.CameraParallelScale = 0.018289574777758052

# Create a new 'Render View'
renderView14 = CreateView('RenderView')
renderView14.ViewSize = [1328, 191]
renderView14.AxesGrid = 'GridAxes3DActor'
renderView14.CenterOfRotation = [0.015262570232152939, 0.009999999776482582, 0.0012499999720603228]
renderView14.StereoType = 'Crystal Eyes'
renderView14.CameraPosition = [0.015262570232152939, 0.009999999776482582, 0.07191549049078692]
renderView14.CameraFocalPoint = [0.015262570232152939, 0.009999999776482582, 0.0012499999720603228]
renderView14.CameraFocalDisk = 1.0
renderView14.CameraParallelScale = 0.018289574777758052

# Create a new 'Render View'
renderView15 = CreateView('RenderView')
renderView15.ViewSize = [1328, 386]
renderView15.AxesGrid = 'GridAxes3DActor'
renderView15.CenterOfRotation = [0.01508551836013794, 0.009999999776482582, 0.0012499999720603228]
renderView15.StereoType = 'Crystal Eyes'
renderView15.CameraPosition = [0.01508551836013794, 0.009999999776482582, 0.07134564562271872]
renderView15.CameraFocalPoint = [0.01508551836013794, 0.009999999776482582, 0.0012499999720603228]
renderView15.CameraFocalDisk = 1.0
renderView15.CameraParallelScale = 0.018142088073148068

# Create a new 'Render View'
renderView16 = CreateView('RenderView')
renderView16.ViewSize = [1328, 280]
renderView16.AxesGrid = 'GridAxes3DActor'
renderView16.CenterOfRotation = [0.01508551836013794, 0.009999999776482582, 0.0012499999720603228]
renderView16.StereoType = 'Crystal Eyes'
renderView16.CameraPosition = [0.01508551836013794, 0.009999999776482582, 0.07134564562271872]
renderView16.CameraFocalPoint = [0.01508551836013794, 0.009999999776482582, 0.0012499999720603228]
renderView16.CameraFocalDisk = 1.0
renderView16.CameraParallelScale = 0.018142088073148068

# Create a new 'Render View'
renderView17 = CreateView('RenderView')
renderView17.ViewSize = [1328, 407]
renderView17.AxesGrid = 'GridAxes3DActor'
renderView17.CenterOfRotation = [0.014400837942957878, 0.009999999776482582, 0.0012499999720603228]
renderView17.StereoType = 'Crystal Eyes'
renderView17.CameraPosition = [0.014722149001452797, 0.009999999776482582, 0.06916108023895852]
renderView17.CameraFocalPoint = [0.014400837942957878, 0.009999999776482582, 0.0012499999720603228]
renderView17.CameraFocalDisk = 1.0
renderView17.CameraParallelScale = 0.01757687767833462

# Create a new 'Render View'
renderView18 = CreateView('RenderView')
renderView18.ViewSize = [1328, 238]
renderView18.AxesGrid = 'GridAxes3DActor'
renderView18.CenterOfRotation = [0.014400837942957878, 0.009999999776482582, 0.0012499999720603228]
renderView18.StereoType = 'Crystal Eyes'
renderView18.CameraPosition = [0.014400837942957878, 0.009999999776482582, 0.06916184035216931]
renderView18.CameraFocalPoint = [0.014400837942957878, 0.009999999776482582, 0.0012499999720603228]
renderView18.CameraFocalDisk = 1.0
renderView18.CameraParallelScale = 0.01757687767833462

# Create a new 'Render View'
renderView19 = CreateView('RenderView')
renderView19.ViewSize = [1328, 405]
renderView19.AxesGrid = 'GridAxes3DActor'
renderView19.CenterOfRotation = [0.01485873106867075, 0.009999999776482582, 0.0012499999720603228]
renderView19.StereoType = 'Crystal Eyes'
renderView19.CameraPosition = [0.01485873106867075, 0.008834662441257385, 0.07060895375466449]
renderView19.CameraFocalPoint = [0.01485873106867075, 0.009999999776482582, 0.0012499999720603228]
renderView19.CameraViewUp = [0.0, 0.9998588839790777, 0.016799170459079988]
renderView19.CameraFocalDisk = 1.0
renderView19.CameraParallelScale = 0.017953951777558163

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [1147, 364]
renderView2.InteractionMode = '2D'
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.OrientationAxesVisibility = 0
renderView2.CenterOfRotation = [0.014576845802366734, 0.009999999776482582, 0.0012499999720603228]
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [0.014790667341474446, 0.009562105861046083, 0.05076833094451296]
renderView2.CameraFocalPoint = [0.014790667341474446, 0.009562105861046083, -0.017701782097909964]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 0.012878338916953409

# Create a new 'Render View'
renderView20 = CreateView('RenderView')
renderView20.ViewSize = [1328, 242]
renderView20.AxesGrid = 'GridAxes3DActor'
renderView20.CenterOfRotation = [0.01485873106867075, 0.009999999776482582, 0.0012499999720603228]
renderView20.StereoType = 'Crystal Eyes'
renderView20.CameraPosition = [0.01485873106867075, 0.009999999776482582, 0.07061874279562814]
renderView20.CameraFocalPoint = [0.01485873106867075, 0.009999999776482582, 0.0012499999720603228]
renderView20.CameraFocalDisk = 1.0
renderView20.CameraParallelScale = 0.017953951777558163

# Create a new 'Render View'
renderView21 = CreateView('RenderView')
renderView21.ViewSize = [1147, 398]
renderView21.AxesGrid = 'GridAxes3DActor'
renderView21.CenterOfRotation = [0.015275806188583374, 0.009999999776482582, 0.0012499999720603228]
renderView21.StereoType = 'Crystal Eyes'
renderView21.CameraPosition = [0.015275806188583374, 0.009999999776482582, 0.07195817205253946]
renderView21.CameraFocalPoint = [0.015275806188583374, 0.009999999776482582, 0.0012499999720603228]
renderView21.CameraFocalDisk = 1.0
renderView21.CameraParallelScale = 0.018300621578814326

# Create a new 'Render View'
renderView22 = CreateView('RenderView')
renderView22.ViewSize = [1147, 256]
renderView22.AxesGrid = 'GridAxes3DActor'
renderView22.CenterOfRotation = [0.015275806188583374, 0.009999999776482582, 0.0012499999720603228]
renderView22.StereoType = 'Crystal Eyes'
renderView22.CameraPosition = [0.015275806188583374, 0.009999999776482582, 0.07195817205253946]
renderView22.CameraFocalPoint = [0.015275806188583374, 0.009999999776482582, 0.0012499999720603228]
renderView22.CameraFocalDisk = 1.0
renderView22.CameraParallelScale = 0.018300621578814326

# Create a new 'Render View'
renderView23 = CreateView('RenderView')
renderView23.ViewSize = [1147, 344]
renderView23.AxesGrid = 'GridAxes3DActor'
renderView23.OrientationAxesVisibility = 0
renderView23.CenterOfRotation = [1.495929479598999, 1.0, 0.125]
renderView23.StereoType = 'Crystal Eyes'
renderView23.CameraPosition = [1.495929479598999, 1.0, 4.822551683074667]
renderView23.CameraFocalPoint = [1.495929479598999, 1.0, -2.146513128928847]
renderView23.CameraFocalDisk = 1.0
renderView23.CameraParallelScale = 1.8037266999003292

# Create a new 'Render View'
renderView24 = CreateView('RenderView')
renderView24.ViewSize = [1147, 331]
renderView24.AxesGrid = 'GridAxes3DActor'
renderView24.CenterOfRotation = [1.504248857498169, 1.0, 0.125]
renderView24.StereoType = 'Crystal Eyes'
renderView24.CameraPosition = [1.504248857498169, 1.0, 7.120746321463815]
renderView24.CameraFocalPoint = [1.504248857498169, 1.0, 0.125]
renderView24.CameraFocalDisk = 1.0
renderView24.CameraParallelScale = 1.8106323827007367

# Create a new 'Render View'
renderView25 = CreateView('RenderView')
renderView25.ViewSize = [1147, 336]
renderView25.AxesGrid = 'GridAxes3DActor'
renderView25.CenterOfRotation = [1.4835350513458252, 1.0, 0.125]
renderView25.StereoType = 'Crystal Eyes'
renderView25.CameraPosition = [1.4835350513458252, 1.0, 7.054399997771284]
renderView25.CameraFocalPoint = [1.4835350513458252, 1.0, 0.125]
renderView25.CameraFocalDisk = 1.0
renderView25.CameraParallelScale = 1.793460690556573

# Create a new 'Render View'
renderView26 = CreateView('RenderView')
renderView26.ViewSize = [1328, 257]
renderView26.AxesGrid = 'GridAxes3DActor'
renderView26.CenterOfRotation = [1.537266731262207, 1.0, 0.125]
renderView26.StereoType = 'Crystal Eyes'
renderView26.CameraPosition = [1.537266731262207, 1.0, 7.2270857580548356]
renderView26.CameraFocalPoint = [1.537266731262207, 1.0, 0.125]
renderView26.CameraFocalDisk = 1.0
renderView26.CameraParallelScale = 1.8381550541359646

# Create a new 'Render View'
renderView27 = CreateView('RenderView')
renderView27.ViewSize = [1328, 261]
renderView27.AxesGrid = 'GridAxes3DActor'
renderView27.CenterOfRotation = [1.52731454372406, 1.0, 0.125]
renderView27.StereoType = 'Crystal Eyes'
renderView27.CameraPosition = [1.52731454372406, 1.0, 7.194959161938187]
renderView27.CameraFocalPoint = [1.52731454372406, 1.0, 0.125]
renderView27.CameraFocalDisk = 1.0
renderView27.CameraParallelScale = 1.8298400792066594

# Create a new 'Render View'
renderView28 = CreateView('RenderView')
renderView28.ViewSize = [1328, 260]
renderView28.AxesGrid = 'GridAxes3DActor'
renderView28.CenterOfRotation = [1.5239006280899048, 1.0, 0.125]
renderView28.StereoType = 'Crystal Eyes'
renderView28.CameraPosition = [1.5239006280899048, 1.0, 7.183953290778913]
renderView28.CameraFocalPoint = [1.5239006280899048, 1.0, 0.125]
renderView28.CameraFocalDisk = 1.0
renderView28.CameraParallelScale = 1.8269915501426945

# Create a new 'Render View'
renderView29 = CreateView('RenderView')
renderView29.ViewSize = [1160, 327]
renderView29.AxesGrid = 'GridAxes3DActor'
renderView29.CenterOfRotation = [1.4998737573623657, 1.0, 0.125]
renderView29.StereoType = 'Crystal Eyes'
renderView29.CameraPosition = [1.4998737573623657, 1.0, 7.106708973061645]
renderView29.CameraFocalPoint = [1.4998737573623657, 1.0, 0.125]
renderView29.CameraFocalDisk = 1.0
renderView29.CameraParallelScale = 1.8069992495915157

# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.ViewSize = [1147, 331]
renderView3.AxesGrid = 'GridAxes3DActor'
renderView3.CenterOfRotation = [0.014550496824085712, 0.009999999776482582, 0.0012499999720603228]
renderView3.StereoType = 'Crystal Eyes'
renderView3.CameraPosition = [0.014550496824085712, 0.008851165029822104, 0.06962674693526946]
renderView3.CameraFocalPoint = [0.014550496824085712, 0.009999999776482582, 0.0012499999720603228]
renderView3.CameraViewUp = [0.0, 0.9998588839790777, 0.01679917045908]
renderView3.CameraFocalDisk = 1.0
renderView3.CameraParallelScale = 0.017699702067761787

# Create a new 'Render View'
renderView30 = CreateView('RenderView')
renderView30.ViewSize = [1328, 386]
renderView30.AxesGrid = 'GridAxes3DActor'
renderView30.CenterOfRotation = [1.5133917331695557, 1.0, 0.125]
renderView30.StereoType = 'Crystal Eyes'
renderView30.CameraPosition = [1.5133917331695557, 1.0, 7.150121665763787]
renderView30.CameraFocalPoint = [1.5133917331695557, 1.0, 0.125]
renderView30.CameraFocalDisk = 1.0
renderView30.CameraParallelScale = 1.818235281262013

# Create a new 'Render View'
renderView31 = CreateView('RenderView')
renderView31.ViewSize = [1328, 407]
renderView31.AxesGrid = 'GridAxes3DActor'
renderView31.CenterOfRotation = [1.4868937730789185, 1.0, 0.125]
renderView31.StereoType = 'Crystal Eyes'
renderView31.CameraPosition = [1.4868937730789185, 1.0, 7.065138376263855]
renderView31.CameraFocalPoint = [1.4868937730789185, 1.0, 0.125]
renderView31.CameraFocalDisk = 1.0
renderView31.CameraParallelScale = 1.7962399874239696

# Create a new 'Render View'
renderView32 = CreateView('RenderView')
renderView32.ViewSize = [1328, 405]
renderView32.AxesGrid = 'GridAxes3DActor'
renderView32.CenterOfRotation = [1.5356347560882568, 1.0, 0.125]
renderView32.StereoType = 'Crystal Eyes'
renderView32.CameraPosition = [1.5356347560882568, 1.0, 7.221813276691123]
renderView32.CameraFocalPoint = [1.5356347560882568, 1.0, 0.125]
renderView32.CameraFocalDisk = 1.0
renderView32.CameraParallelScale = 1.8367904355440878

# Create a new 'Render View'
renderView33 = CreateView('RenderView')
renderView33.ViewSize = [1147, 398]
renderView33.AxesGrid = 'GridAxes3DActor'
renderView33.CenterOfRotation = [1.543668270111084, 1.0, 0.125]
renderView33.StereoType = 'Crystal Eyes'
renderView33.CameraPosition = [1.543668270111084, 1.0, 7.247783652712271]
renderView33.CameraFocalPoint = [1.543668270111084, 1.0, 0.125]
renderView33.CameraFocalDisk = 1.0
renderView33.CameraParallelScale = 1.8435120634668347

# Create a new 'Render View'
renderView4 = CreateView('RenderView')
renderView4.ViewSize = [1147, 390]
renderView4.AxesGrid = 'GridAxes3DActor'
renderView4.CenterOfRotation = [0.014550496824085712, 0.009999999776482582, 0.0012499999720603228]
renderView4.StereoType = 'Crystal Eyes'
renderView4.CameraPosition = [0.014550496824085712, 0.009999999776482582, 0.06963639735155287]
renderView4.CameraFocalPoint = [0.014550496824085712, 0.009999999776482582, 0.0012499999720603228]
renderView4.CameraFocalDisk = 1.0
renderView4.CameraParallelScale = 0.017699702067761787

# Create a new 'Render View'
renderView5 = CreateView('RenderView')
renderView5.ViewSize = [1160, 328]
renderView5.AxesGrid = 'GridAxes3DActor'
renderView5.CenterOfRotation = [0.015122383832931519, 0.009999999776482582, 0.0012499999720603228]
renderView5.StereoType = 'Crystal Eyes'
renderView5.CameraPosition = [0.015122383832931519, 0.009999999776482582, 0.0714641296981597]
renderView5.CameraFocalPoint = [0.015122383832931519, 0.009999999776482582, 0.0012499999720603228]
renderView5.CameraFocalDisk = 1.0
renderView5.CameraParallelScale = 0.018172754008413557

# Create a new 'Render View'
renderView6 = CreateView('RenderView')
renderView6.ViewSize = [1160, 397]
renderView6.AxesGrid = 'GridAxes3DActor'
renderView6.CenterOfRotation = [0.015122383832931519, 0.009999999776482582, 0.0012499999720603228]
renderView6.StereoType = 'Crystal Eyes'
renderView6.CameraPosition = [0.015122383832931519, 0.009999999776482582, 0.0714641296981597]
renderView6.CameraFocalPoint = [0.015122383832931519, 0.009999999776482582, 0.0012499999720603228]
renderView6.CameraFocalDisk = 1.0
renderView6.CameraParallelScale = 0.018172754008413557

# Create a new 'Render View'
renderView7 = CreateView('RenderView')
renderView7.ViewSize = [1147, 336]
renderView7.AxesGrid = 'GridAxes3DActor'
renderView7.CenterOfRotation = [0.01460856944322586, 0.009999999776482582, 0.0012499999720603228]
renderView7.StereoType = 'Crystal Eyes'
renderView7.CameraPosition = [0.01460856944322586, 0.009999999776482582, 0.06982096992444375]
renderView7.CameraFocalPoint = [0.01460856944322586, 0.009999999776482582, 0.0012499999720603228]
renderView7.CameraFocalDisk = 1.0
renderView7.CameraParallelScale = 0.017747472964829522

# Create a new 'Render View'
renderView8 = CreateView('RenderView')
renderView8.ViewSize = [1147, 380]
renderView8.AxesGrid = 'GridAxes3DActor'
renderView8.CenterOfRotation = [0.01460856944322586, 0.009999999776482582, 0.0012499999720603228]
renderView8.StereoType = 'Crystal Eyes'
renderView8.CameraPosition = [0.01460856944322586, 0.009999999776482582, 0.06982096992444375]
renderView8.CameraFocalPoint = [0.01460856944322586, 0.009999999776482582, 0.0012499999720603228]
renderView8.CameraFocalDisk = 1.0
renderView8.CameraParallelScale = 0.017747472964829522

# Create a new 'Render View'
renderView9 = CreateView('RenderView')
renderView9.ViewSize = [1328, 257]
renderView9.AxesGrid = 'GridAxes3DActor'
renderView9.CenterOfRotation = [0.014669972471892834, 0.009999999776482582, 0.0012499999720603228]
renderView9.StereoType = 'Crystal Eyes'
renderView9.CameraPosition = [0.014669972471892834, 0.009999999776482582, 0.07001638501476622]
renderView9.CameraFocalPoint = [0.014669972471892834, 0.009999999776482582, 0.0012499999720603228]
renderView9.CameraFocalDisk = 1.0
renderView9.CameraParallelScale = 0.017798050111905404

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object '0'
a0 = CreateLayout(name='0')
a0.SplitVertical(0, 0.655637)
a0.SplitVertical(1, 0.500000)
a0.AssignView(3, renderView1)
a0.AssignView(4, renderView23)
a0.AssignView(2, renderView2)
a0.SetSize(1147, 1054)

# create new layout object '1'
a1 = CreateLayout(name='1')
a1.SplitVertical(0, 0.633578)
a1.SplitVertical(1, 0.500000)
a1.AssignView(3, renderView3)
a1.AssignView(4, renderView24)
a1.AssignView(2, renderView4)
a1.SetSize(1147, 1054)

# create new layout object '10'
a10 = CreateLayout(name='10')
a10.SplitVertical(0, 0.750000)
a10.SplitVertical(1, 0.500000)
a10.AssignView(3, renderView21)
a10.AssignView(4, renderView33)
a10.AssignView(2, renderView22)
a10.SetSize(1147, 1054)

# create new layout object '2'
a2 = CreateLayout(name='2')
a2.SplitVertical(0, 0.627451)
a2.SplitVertical(1, 0.500000)
a2.AssignView(3, renderView5)
a2.AssignView(4, renderView29)
a2.AssignView(2, renderView6)
a2.SetSize(1160, 1054)

# create new layout object '3'
a3 = CreateLayout(name='3')
a3.SplitVertical(0, 0.642157)
a3.SplitVertical(1, 0.500000)
a3.AssignView(3, renderView7)
a3.AssignView(4, renderView25)
a3.AssignView(2, renderView8)
a3.SetSize(1147, 1054)

# create new layout object '4'
a4 = CreateLayout(name='4')
a4.SplitVertical(0, 0.715686)
a4.SplitVertical(1, 0.500000)
a4.AssignView(3, renderView9)
a4.AssignView(4, renderView26)
a4.AssignView(2, renderView10)
a4.SetSize(1328, 714)

# create new layout object '5'
a5 = CreateLayout(name='5')
a5.SplitVertical(0, 0.725490)
a5.SplitVertical(1, 0.500000)
a5.AssignView(3, renderView11)
a5.AssignView(4, renderView27)
a5.AssignView(2, renderView12)
a5.SetSize(1328, 714)

# create new layout object '6'
a6 = CreateLayout(name='6')
a6.SplitVertical(0, 0.724265)
a6.SplitVertical(1, 0.500000)
a6.AssignView(3, renderView13)
a6.AssignView(4, renderView28)
a6.AssignView(2, renderView14)
a6.SetSize(1328, 714)

# create new layout object '7'
a7 = CreateLayout(name='7')
a7.SplitVertical(0, 0.729167)
a7.SplitVertical(1, 0.500000)
a7.AssignView(3, renderView15)
a7.AssignView(4, renderView30)
a7.AssignView(2, renderView16)
a7.SetSize(1328, 1054)

# create new layout object '8'
a8 = CreateLayout(name='8')
a8.SplitVertical(0, 0.765571)
a8.SplitVertical(1, 0.500000)
a8.AssignView(3, renderView17)
a8.AssignView(4, renderView31)
a8.AssignView(2, renderView18)
a8.SetSize(1328, 1054)

# create new layout object '9'
a9 = CreateLayout(name='9')
a9.SplitVertical(0, 0.762111)
a9.SplitVertical(1, 0.500000)
a9.AssignView(3, renderView19)
a9.AssignView(4, renderView32)
a9.AssignView(2, renderView20)
a9.SetSize(1328, 1054)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView5)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Legacy VTK Reader'
a10_1 = LegacyVTKReader(registrationName='10', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth10_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth10_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth10_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth10_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth10_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth10_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth10_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth10_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth10_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth10_theta1.0/output_7901.vtk'])

# create a new 'Legacy VTK Reader'
a4_1 = LegacyVTKReader(registrationName='4', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_8000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth4_theta1.0/output_8001.vtk'])

# create a new 'Legacy VTK Reader'
a2_1 = LegacyVTKReader(registrationName='2', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth2_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth2_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth2_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth2_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth2_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth2_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth2_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth2_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth2_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth2_theta1.0/output_7501.vtk'])

# create a new 'Legacy VTK Reader'
a5_1 = LegacyVTKReader(registrationName='5', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth5_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth5_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth5_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth5_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth5_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth5_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth5_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth5_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth5_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth5_theta1.0/output_7801.vtk'])

# create a new 'Legacy VTK Reader'
a3_1 = LegacyVTKReader(registrationName='3', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth3_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth3_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth3_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth3_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth3_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth3_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth3_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth3_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth3_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth3_theta1.0/output_7201.vtk'])

# create a new 'XML Structured Grid Reader'
net_7 = XMLStructuredGridReader(registrationName='net_7', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom7_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom7_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom7_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom7_tx3.8e+04/deformed_3000.vts'])
net_7.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net_7.TimeArray = 'None'

# create a new 'XML Structured Grid Reader'
net_6 = XMLStructuredGridReader(registrationName='net_6', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom6_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom6_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom6_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom6_tx3.8e+04/deformed_2600.vts'])
net_6.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net_6.TimeArray = 'None'

# create a new 'Legacy VTK Reader'
a0_1 = LegacyVTKReader(registrationName='0', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth0_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth0_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth0_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth0_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth0_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth0_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth0_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth0_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth0_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth0_theta1.0/output_7301.vtk'])

# create a new 'Legacy VTK Reader'
a6_1 = LegacyVTKReader(registrationName='6', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth6_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth6_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth6_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth6_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth6_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth6_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth6_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth6_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth6_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth6_theta1.0/output_7501.vtk'])

# create a new 'XML Structured Grid Reader'
net_2 = XMLStructuredGridReader(registrationName='net_2', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom2_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom2_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom2_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom2_tx3.8e+04/deformed_3000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom2_tx3.8e+04/deformed_3400.vts'])
net_2.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net_2.TimeArray = 'None'

# create a new 'Legacy VTK Reader'
a7_1 = LegacyVTKReader(registrationName='7', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth7_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth7_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth7_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth7_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth7_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth7_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth7_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth7_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth7_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth7_theta1.0/output_7701.vtk'])

# create a new 'Legacy VTK Reader'
a8_1 = LegacyVTKReader(registrationName='8', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth8_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth8_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth8_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth8_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth8_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth8_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth8_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth8_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth8_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth8_theta1.0/output_7401.vtk'])

# create a new 'Legacy VTK Reader'
a9_1 = LegacyVTKReader(registrationName='9', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth9_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth9_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth9_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth9_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth9_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth9_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth9_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth9_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth9_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth9_theta1.0/output_7701.vtk'])

# create a new 'XML Structured Grid Reader'
net0 = XMLStructuredGridReader(registrationName='net 0', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom0_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom0_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom0_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom0_tx3.8e+04/deformed_3000.vts'])
net0.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net0.TimeArray = 'None'

# create a new 'Legacy VTK Reader'
a1_1 = LegacyVTKReader(registrationName='1', FileNames=['/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth1_theta1.0/output_0.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth1_theta1.0/output_1.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth1_theta1.0/output_1000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth1_theta1.0/output_2000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth1_theta1.0/output_3000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth1_theta1.0/output_4000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth1_theta1.0/output_5000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth1_theta1.0/output_6000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth1_theta1.0/output_7000.vtk', '/home/shguan/dem_hyperelasticity/torch_Lagrange/3d_degraded_panel_random_tx/fem_panel_tx_randomHealthField/panel_3D_degraded_K5e+05_mu6.2e+03_randomHealth1_theta1.0/output_7501.vtk'])

# create a new 'XML Structured Grid Reader'
net_1 = XMLStructuredGridReader(registrationName='net_1', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom1_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom1_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom1_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom1_tx3.8e+04/deformed_2900.vts'])
net_1.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net_1.TimeArray = 'None'

# create a new 'XML Structured Grid Reader'
net_9 = XMLStructuredGridReader(registrationName='net_9', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom9_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom9_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom9_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom9_tx3.8e+04/deformed_3000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom9_tx3.8e+04/deformed_3400.vts'])
net_9.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net_9.TimeArray = 'None'

# create a new 'XML Structured Grid Reader'
net_4 = XMLStructuredGridReader(registrationName='net_4', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom4_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom4_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom4_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom4_tx3.8e+04/deformed_2100.vts'])
net_4.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net_4.TimeArray = 'None'

# create a new 'XML Structured Grid Reader'
net_3 = XMLStructuredGridReader(registrationName='net_3', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom3_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom3_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom3_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom3_tx3.8e+04/deformed_2800.vts'])
net_3.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net_3.TimeArray = 'None'

# create a new 'XML Structured Grid Reader'
net_10 = XMLStructuredGridReader(registrationName='net_10', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom10_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom10_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom10_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom10_tx3.8e+04/deformed_2700.vts'])
net_10.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net_10.TimeArray = 'None'

# create a new 'XML Structured Grid Reader'
net_8 = XMLStructuredGridReader(registrationName='net_8', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom8_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom8_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom8_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom8_tx3.8e+04/deformed_2500.vts'])
net_8.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net_8.TimeArray = 'None'

# create a new 'XML Structured Grid Reader'
net_5 = XMLStructuredGridReader(registrationName='net_5', FileName=['/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom5_tx3.8e+04/deformed_0.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom5_tx3.8e+04/deformed_1000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom5_tx3.8e+04/deformed_2000.vts', '/home/shguan/dem_hyperelasticity/degraded_random/simu_normalNet_random_tx/degraded_simpson_25x25x4_theta1.0_phi0.0_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+06_helthRandom5_tx3.8e+04/deformed_2400.vts'])
net_5.PointArrayStatus = ['displacement', 'S-VonMises', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'sig_v', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33', 'eps_v', 'E11cauchy', 'E12cauchy', 'E13cauchy', 'E22cauchy', 'E23cauchy', 'E33cauchy', 'eps_v_cauchy', 'pk1_11', 'pk1_12', 'pk1_13', 'pk1_22', 'pk1_23', 'pk1_33', 'pk2_11', 'pk2_12', 'pk2_13', 'pk2_22', 'pk2_23', 'pk2_33', 'random_field']
net_5.TimeArray = 'None'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from a0_1
a0_1Display = Show(a0_1, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'pk1'
pk1LUT = GetColorTransferFunction('pk1')
pk1LUT.AutomaticRescaleRangeMode = 'Never'
pk1LUT.RGBPoints = [15000.0, 0.02, 0.3813, 0.9981, 15833.333333333334, 0.02000006, 0.424267768, 0.96906969, 16666.666666666668, 0.02, 0.467233763, 0.940033043, 17500.0, 0.02, 0.5102, 0.911, 18333.333333333332, 0.02000006, 0.546401494, 0.872669438, 19166.666666666664, 0.02, 0.582600362, 0.83433295, 20000.0, 0.02, 0.6188, 0.796, 20833.333333333332, 0.02000006, 0.652535156, 0.749802434, 21666.666666666664, 0.02, 0.686267004, 0.703599538, 22500.0, 0.02, 0.72, 0.6574, 23333.333333333332, 0.02000006, 0.757035456, 0.603735359, 24166.666666666668, 0.02, 0.794067037, 0.55006613, 25000.0, 0.02, 0.8311, 0.4964, 25833.333333333336, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 26666.666666666664, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 27500.0, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 28333.333333333332, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 29166.666666666664, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 30000.0, 0.6439, 0.9773, 0.0469, 30833.333333333336, 0.762401813, 0.984669591, 0.034600153, 31666.666666666664, 0.880901185, 0.992033407, 0.022299877, 32500.0, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 33333.333333333336, 0.999402998, 0.955036376, 0.079066628, 34166.66666666667, 0.9994, 0.910666223, 0.148134024, 35000.0, 0.9994, 0.8663, 0.2172, 35833.33333333333, 0.999269665, 0.818035981, 0.217200652, 36666.66666666667, 0.999133332, 0.769766184, 0.2172, 37500.0, 0.999, 0.7215, 0.2172, 38333.33333333333, 0.99913633, 0.673435546, 0.217200652, 39166.66666666667, 0.999266668, 0.625366186, 0.2172, 40000.0, 0.9994, 0.5773, 0.2172, 40833.333333333336, 0.999402998, 0.521068455, 0.217200652, 41666.666666666664, 0.9994, 0.464832771, 0.2172, 42500.0, 0.9994, 0.4086, 0.2172, 43333.33333333333, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 44166.66666666667, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 45000.0, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 45833.33333333333, 0.949903037, 0.116867171, 0.252900603, 46666.66666666667, 0.903199533, 0.078432949, 0.291800389, 47500.0, 0.8565, 0.04, 0.3307, 48333.33333333333, 0.798902627, 0.04333345, 0.358434298, 49166.666666666664, 0.741299424, 0.0466667, 0.386166944, 50000.0, 0.6837, 0.05, 0.4139]
pk1LUT.ColorSpace = 'RGB'
pk1LUT.NanColor = [1.0, 0.0, 0.0]
pk1LUT.ScalarRangeInitialized = 1.0
pk1LUT.VectorMode = 'Component'

# get opacity transfer function/opacity map for 'pk1'
pk1PWF = GetOpacityTransferFunction('pk1')
pk1PWF.Points = [15000.0, 0.0, 0.5, 0.0, 50000.0, 1.0, 0.5, 0.0]
pk1PWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
a0_1Display.Representation = 'Surface'
a0_1Display.ColorArrayName = ['POINTS', 'pk1']
a0_1Display.LookupTable = pk1LUT
a0_1Display.SelectTCoordArray = 'None'
a0_1Display.SelectNormalArray = 'None'
a0_1Display.SelectTangentArray = 'None'
a0_1Display.EdgeColor = [0.8705882352941177, 0.8666666666666667, 0.8549019607843137]
a0_1Display.OSPRayScaleArray = 'Eps'
a0_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a0_1Display.SelectOrientationVectors = 'None'
a0_1Display.ScaleFactor = 0.0019999999552965165
a0_1Display.SelectScaleArray = 'Eps_v'
a0_1Display.GlyphType = 'Arrow'
a0_1Display.GlyphTableIndexArray = 'Eps'
a0_1Display.GaussianRadius = 9.999999776482583e-05
a0_1Display.SetScaleArray = ['POINTS', 'Eps']
a0_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a0_1Display.OpacityArray = ['POINTS', 'Eps']
a0_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a0_1Display.DataAxesGrid = 'GridAxesRepresentation'
a0_1Display.PolarAxes = 'PolarAxesRepresentation'
a0_1Display.ScalarOpacityFunction = pk1PWF
a0_1Display.ScalarOpacityUnitDistance = 0.0020726890923079408
a0_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a0_1Display.ScaleTransferFunction.Points = [-5.960464477539063e-08, 0.0, 0.5, 0.0, 5.960464477539063e-08, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a0_1Display.OpacityTransferFunction.Points = [-5.960464477539063e-08, 0.0, 0.5, 0.0, 5.960464477539063e-08, 1.0, 0.5, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView10'
# ----------------------------------------------------------------

# show data from a4_1
a4_1Display = Show(a4_1, renderView10, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'health'
healthLUT = GetColorTransferFunction('health')
healthLUT.RGBPoints = [0.20146505716446095, 0.23137254902, 0.298039215686, 0.752941176471, 0.6003666562650483, 0.865, 0.865, 0.865, 0.9992682553656356, 0.705882352941, 0.0156862745098, 0.149019607843]
healthLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'health'
healthPWF = GetOpacityTransferFunction('health')
healthPWF.Points = [0.20146505716446095, 0.0, 0.5, 0.0, 0.9992682553656356, 1.0, 0.5, 0.0]
healthPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
a4_1Display.Representation = 'Surface'
a4_1Display.ColorArrayName = ['POINTS', 'health']
a4_1Display.LookupTable = healthLUT
a4_1Display.SelectTCoordArray = 'None'
a4_1Display.SelectNormalArray = 'None'
a4_1Display.SelectTangentArray = 'None'
a4_1Display.OSPRayScaleArray = 'Eps'
a4_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a4_1Display.SelectOrientationVectors = 'None'
a4_1Display.ScaleFactor = 0.002933994494378567
a4_1Display.SelectScaleArray = 'Eps_v'
a4_1Display.GlyphType = 'Arrow'
a4_1Display.GlyphTableIndexArray = 'Eps'
a4_1Display.GaussianRadius = 0.00014669972471892834
a4_1Display.SetScaleArray = ['POINTS', 'Eps']
a4_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a4_1Display.OpacityArray = ['POINTS', 'Eps']
a4_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a4_1Display.DataAxesGrid = 'GridAxesRepresentation'
a4_1Display.PolarAxes = 'PolarAxesRepresentation'
a4_1Display.ScalarOpacityFunction = healthPWF
a4_1Display.ScalarOpacityUnitDistance = 0.0025983743994128345
a4_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a4_1Display.ScaleTransferFunction.Points = [0.01342993974685669, 0.0, 0.5, 0.0, 0.7344027876853944, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a4_1Display.OpacityTransferFunction.Points = [0.01342993974685669, 0.0, 0.5, 0.0, 0.7344027876853944, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView10
healthLUTColorBar = GetScalarBar(healthLUT, renderView10)
healthLUTColorBar.Title = 'health'
healthLUTColorBar.ComponentTitle = ''

# set color bar visibility
healthLUTColorBar.Visibility = 1

# show color legend
a4_1Display.SetScalarBarVisibility(renderView10, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView11'
# ----------------------------------------------------------------

# show data from a5_1
a5_1Display = Show(a5_1, renderView11, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a5_1Display.Representation = 'Surface'
a5_1Display.ColorArrayName = ['POINTS', 'pk1']
a5_1Display.LookupTable = pk1LUT
a5_1Display.SelectTCoordArray = 'None'
a5_1Display.SelectNormalArray = 'None'
a5_1Display.SelectTangentArray = 'None'
a5_1Display.OSPRayScaleArray = 'Eps'
a5_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a5_1Display.SelectOrientationVectors = 'None'
a5_1Display.ScaleFactor = 0.0029324185103178028
a5_1Display.SelectScaleArray = 'Eps_v'
a5_1Display.GlyphType = 'Arrow'
a5_1Display.GlyphTableIndexArray = 'Eps'
a5_1Display.GaussianRadius = 0.0001466209255158901
a5_1Display.SetScaleArray = ['POINTS', 'Eps']
a5_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a5_1Display.OpacityArray = ['POINTS', 'Eps']
a5_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a5_1Display.DataAxesGrid = 'GridAxesRepresentation'
a5_1Display.PolarAxes = 'PolarAxesRepresentation'
a5_1Display.ScalarOpacityFunction = pk1PWF
a5_1Display.ScalarOpacityUnitDistance = 0.002597426263514604
a5_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a5_1Display.ScaleTransferFunction.Points = [0.018478840589523315, 0.0, 0.5, 0.0, 0.6727590262889862, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a5_1Display.OpacityTransferFunction.Points = [0.018478840589523315, 0.0, 0.5, 0.0, 0.6727590262889862, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1LUT in view renderView11
pk1LUTColorBar = GetScalarBar(pk1LUT, renderView11)
pk1LUTColorBar.Title = 'pk1'
pk1LUTColorBar.ComponentTitle = '0'

# set color bar visibility
pk1LUTColorBar.Visibility = 1

# show color legend
a5_1Display.SetScalarBarVisibility(renderView11, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView12'
# ----------------------------------------------------------------

# show data from a5_1
a5_1Display_1 = Show(a5_1, renderView12, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a5_1Display_1.Representation = 'Surface'
a5_1Display_1.ColorArrayName = ['POINTS', 'health']
a5_1Display_1.LookupTable = healthLUT
a5_1Display_1.SelectTCoordArray = 'None'
a5_1Display_1.SelectNormalArray = 'None'
a5_1Display_1.SelectTangentArray = 'None'
a5_1Display_1.OSPRayScaleArray = 'Eps'
a5_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a5_1Display_1.SelectOrientationVectors = 'None'
a5_1Display_1.ScaleFactor = 0.0029324185103178028
a5_1Display_1.SelectScaleArray = 'Eps_v'
a5_1Display_1.GlyphType = 'Arrow'
a5_1Display_1.GlyphTableIndexArray = 'Eps'
a5_1Display_1.GaussianRadius = 0.0001466209255158901
a5_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a5_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a5_1Display_1.OpacityArray = ['POINTS', 'Eps']
a5_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a5_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a5_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a5_1Display_1.ScalarOpacityFunction = healthPWF
a5_1Display_1.ScalarOpacityUnitDistance = 0.002597426263514604
a5_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a5_1Display_1.ScaleTransferFunction.Points = [0.018478840589523315, 0.0, 0.5, 0.0, 0.6727590262889862, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a5_1Display_1.OpacityTransferFunction.Points = [0.018478840589523315, 0.0, 0.5, 0.0, 0.6727590262889862, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView12
healthLUTColorBar_1 = GetScalarBar(healthLUT, renderView12)
healthLUTColorBar_1.Title = 'health'
healthLUTColorBar_1.ComponentTitle = ''

# set color bar visibility
healthLUTColorBar_1.Visibility = 1

# show color legend
a5_1Display_1.SetScalarBarVisibility(renderView12, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView13'
# ----------------------------------------------------------------

# show data from a6_1
a6_1Display = Show(a6_1, renderView13, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a6_1Display.Representation = 'Surface'
a6_1Display.ColorArrayName = ['POINTS', 'pk1']
a6_1Display.LookupTable = pk1LUT
a6_1Display.SelectTCoordArray = 'None'
a6_1Display.SelectNormalArray = 'None'
a6_1Display.SelectTangentArray = 'None'
a6_1Display.OSPRayScaleArray = 'Eps'
a6_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a6_1Display.SelectOrientationVectors = 'None'
a6_1Display.ScaleFactor = 0.0030525140464305878
a6_1Display.SelectScaleArray = 'Eps_v'
a6_1Display.GlyphType = 'Arrow'
a6_1Display.GlyphTableIndexArray = 'Eps'
a6_1Display.GaussianRadius = 0.0001526257023215294
a6_1Display.SetScaleArray = ['POINTS', 'Eps']
a6_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a6_1Display.OpacityArray = ['POINTS', 'Eps']
a6_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a6_1Display.DataAxesGrid = 'GridAxesRepresentation'
a6_1Display.PolarAxes = 'PolarAxesRepresentation'
a6_1Display.ScalarOpacityFunction = pk1PWF
a6_1Display.ScalarOpacityUnitDistance = 0.0026701331089569293
a6_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a6_1Display.ScaleTransferFunction.Points = [0.020862549543380737, 0.0, 0.5, 0.0, 0.7188984751701355, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a6_1Display.OpacityTransferFunction.Points = [0.020862549543380737, 0.0, 0.5, 0.0, 0.7188984751701355, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1LUT in view renderView13
pk1LUTColorBar_1 = GetScalarBar(pk1LUT, renderView13)
pk1LUTColorBar_1.Title = 'pk1'
pk1LUTColorBar_1.ComponentTitle = '0'

# set color bar visibility
pk1LUTColorBar_1.Visibility = 1

# show color legend
a6_1Display.SetScalarBarVisibility(renderView13, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView14'
# ----------------------------------------------------------------

# show data from a6_1
a6_1Display_1 = Show(a6_1, renderView14, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a6_1Display_1.Representation = 'Surface'
a6_1Display_1.ColorArrayName = ['POINTS', 'health']
a6_1Display_1.LookupTable = healthLUT
a6_1Display_1.SelectTCoordArray = 'None'
a6_1Display_1.SelectNormalArray = 'None'
a6_1Display_1.SelectTangentArray = 'None'
a6_1Display_1.OSPRayScaleArray = 'Eps'
a6_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a6_1Display_1.SelectOrientationVectors = 'None'
a6_1Display_1.ScaleFactor = 0.0030525140464305878
a6_1Display_1.SelectScaleArray = 'Eps_v'
a6_1Display_1.GlyphType = 'Arrow'
a6_1Display_1.GlyphTableIndexArray = 'Eps'
a6_1Display_1.GaussianRadius = 0.0001526257023215294
a6_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a6_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a6_1Display_1.OpacityArray = ['POINTS', 'Eps']
a6_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a6_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a6_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a6_1Display_1.ScalarOpacityFunction = healthPWF
a6_1Display_1.ScalarOpacityUnitDistance = 0.0026701331089569293
a6_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a6_1Display_1.ScaleTransferFunction.Points = [0.020862549543380737, 0.0, 0.5, 0.0, 0.7188984751701355, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a6_1Display_1.OpacityTransferFunction.Points = [0.020862549543380737, 0.0, 0.5, 0.0, 0.7188984751701355, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView14
healthLUTColorBar_2 = GetScalarBar(healthLUT, renderView14)
healthLUTColorBar_2.Title = 'health'
healthLUTColorBar_2.ComponentTitle = ''

# set color bar visibility
healthLUTColorBar_2.Visibility = 1

# show color legend
a6_1Display_1.SetScalarBarVisibility(renderView14, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView15'
# ----------------------------------------------------------------

# show data from a7_1
a7_1Display = Show(a7_1, renderView15, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a7_1Display.Representation = 'Surface'
a7_1Display.ColorArrayName = ['POINTS', 'pk1']
a7_1Display.LookupTable = pk1LUT
a7_1Display.SelectTCoordArray = 'None'
a7_1Display.SelectNormalArray = 'None'
a7_1Display.SelectTangentArray = 'None'
a7_1Display.OSPRayScaleArray = 'Eps'
a7_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a7_1Display.SelectOrientationVectors = 'None'
a7_1Display.ScaleFactor = 0.0030171036720275882
a7_1Display.SelectScaleArray = 'Eps_v'
a7_1Display.GlyphType = 'Arrow'
a7_1Display.GlyphTableIndexArray = 'Eps'
a7_1Display.GaussianRadius = 0.00015085518360137938
a7_1Display.SetScaleArray = ['POINTS', 'Eps']
a7_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a7_1Display.OpacityArray = ['POINTS', 'Eps']
a7_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a7_1Display.DataAxesGrid = 'GridAxesRepresentation'
a7_1Display.PolarAxes = 'PolarAxesRepresentation'
a7_1Display.ScalarOpacityFunction = pk1PWF
a7_1Display.ScalarOpacityUnitDistance = 0.002648601217817012
a7_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a7_1Display.ScaleTransferFunction.Points = [0.016950875520706177, 0.0, 0.5, 0.0, 0.7174374716622488, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a7_1Display.OpacityTransferFunction.Points = [0.016950875520706177, 0.0, 0.5, 0.0, 0.7174374716622488, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1LUT in view renderView15
pk1LUTColorBar_2 = GetScalarBar(pk1LUT, renderView15)
pk1LUTColorBar_2.Title = 'pk1'
pk1LUTColorBar_2.ComponentTitle = '0'

# set color bar visibility
pk1LUTColorBar_2.Visibility = 1

# show color legend
a7_1Display.SetScalarBarVisibility(renderView15, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView16'
# ----------------------------------------------------------------

# show data from a7_1
a7_1Display_1 = Show(a7_1, renderView16, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a7_1Display_1.Representation = 'Surface'
a7_1Display_1.ColorArrayName = ['POINTS', 'health']
a7_1Display_1.LookupTable = healthLUT
a7_1Display_1.SelectTCoordArray = 'None'
a7_1Display_1.SelectNormalArray = 'None'
a7_1Display_1.SelectTangentArray = 'None'
a7_1Display_1.OSPRayScaleArray = 'Eps'
a7_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a7_1Display_1.SelectOrientationVectors = 'None'
a7_1Display_1.ScaleFactor = 0.0030171036720275882
a7_1Display_1.SelectScaleArray = 'Eps_v'
a7_1Display_1.GlyphType = 'Arrow'
a7_1Display_1.GlyphTableIndexArray = 'Eps'
a7_1Display_1.GaussianRadius = 0.00015085518360137938
a7_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a7_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a7_1Display_1.OpacityArray = ['POINTS', 'Eps']
a7_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a7_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a7_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a7_1Display_1.ScalarOpacityFunction = healthPWF
a7_1Display_1.ScalarOpacityUnitDistance = 0.002648601217817012
a7_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a7_1Display_1.ScaleTransferFunction.Points = [0.016950875520706177, 0.0, 0.5, 0.0, 0.7174374716622488, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a7_1Display_1.OpacityTransferFunction.Points = [0.016950875520706177, 0.0, 0.5, 0.0, 0.7174374716622488, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView16
healthLUTColorBar_3 = GetScalarBar(healthLUT, renderView16)
healthLUTColorBar_3.Title = 'health'
healthLUTColorBar_3.ComponentTitle = ''

# set color bar visibility
healthLUTColorBar_3.Visibility = 1

# show color legend
a7_1Display_1.SetScalarBarVisibility(renderView16, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView17'
# ----------------------------------------------------------------

# show data from a8_1
a8_1Display = Show(a8_1, renderView17, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a8_1Display.Representation = 'Surface'
a8_1Display.ColorArrayName = ['POINTS', 'pk1']
a8_1Display.LookupTable = pk1LUT
a8_1Display.SelectTCoordArray = 'None'
a8_1Display.SelectNormalArray = 'None'
a8_1Display.SelectTangentArray = 'None'
a8_1Display.OSPRayScaleArray = 'Eps'
a8_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a8_1Display.SelectOrientationVectors = 'None'
a8_1Display.ScaleFactor = 0.002880167588591576
a8_1Display.SelectScaleArray = 'Eps_v'
a8_1Display.GlyphType = 'Arrow'
a8_1Display.GlyphTableIndexArray = 'Eps'
a8_1Display.GaussianRadius = 0.0001440083794295788
a8_1Display.SetScaleArray = ['POINTS', 'Eps']
a8_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a8_1Display.OpacityArray = ['POINTS', 'Eps']
a8_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a8_1Display.DataAxesGrid = 'GridAxesRepresentation'
a8_1Display.PolarAxes = 'PolarAxesRepresentation'
a8_1Display.ScalarOpacityFunction = pk1PWF
a8_1Display.ScalarOpacityUnitDistance = 0.002566084975254974
a8_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a8_1Display.ScaleTransferFunction.Points = [0.015365809202194214, 0.0, 0.5, 0.0, 0.5557978987693787, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a8_1Display.OpacityTransferFunction.Points = [0.015365809202194214, 0.0, 0.5, 0.0, 0.5557978987693787, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1LUT in view renderView17
pk1LUTColorBar_3 = GetScalarBar(pk1LUT, renderView17)
pk1LUTColorBar_3.Title = 'pk1'
pk1LUTColorBar_3.ComponentTitle = '0'

# set color bar visibility
pk1LUTColorBar_3.Visibility = 1

# show color legend
a8_1Display.SetScalarBarVisibility(renderView17, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView18'
# ----------------------------------------------------------------

# show data from a8_1
a8_1Display_1 = Show(a8_1, renderView18, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a8_1Display_1.Representation = 'Surface'
a8_1Display_1.ColorArrayName = ['POINTS', 'health']
a8_1Display_1.LookupTable = healthLUT
a8_1Display_1.SelectTCoordArray = 'None'
a8_1Display_1.SelectNormalArray = 'None'
a8_1Display_1.SelectTangentArray = 'None'
a8_1Display_1.OSPRayScaleArray = 'Eps'
a8_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a8_1Display_1.SelectOrientationVectors = 'None'
a8_1Display_1.ScaleFactor = 0.002880167588591576
a8_1Display_1.SelectScaleArray = 'Eps_v'
a8_1Display_1.GlyphType = 'Arrow'
a8_1Display_1.GlyphTableIndexArray = 'Eps'
a8_1Display_1.GaussianRadius = 0.0001440083794295788
a8_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a8_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a8_1Display_1.OpacityArray = ['POINTS', 'Eps']
a8_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a8_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a8_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a8_1Display_1.ScalarOpacityFunction = healthPWF
a8_1Display_1.ScalarOpacityUnitDistance = 0.002566084975254974
a8_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a8_1Display_1.ScaleTransferFunction.Points = [0.015365809202194214, 0.0, 0.5, 0.0, 0.5557978987693787, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a8_1Display_1.OpacityTransferFunction.Points = [0.015365809202194214, 0.0, 0.5, 0.0, 0.5557978987693787, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView18
healthLUTColorBar_4 = GetScalarBar(healthLUT, renderView18)
healthLUTColorBar_4.Title = 'health'
healthLUTColorBar_4.ComponentTitle = ''

# set color bar visibility
healthLUTColorBar_4.Visibility = 1

# show color legend
a8_1Display_1.SetScalarBarVisibility(renderView18, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView19'
# ----------------------------------------------------------------

# show data from a9_1
a9_1Display = Show(a9_1, renderView19, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a9_1Display.Representation = 'Surface'
a9_1Display.ColorArrayName = ['POINTS', 'pk1']
a9_1Display.LookupTable = pk1LUT
a9_1Display.SelectTCoordArray = 'None'
a9_1Display.SelectNormalArray = 'None'
a9_1Display.SelectTangentArray = 'None'
a9_1Display.OSPRayScaleArray = 'Eps'
a9_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a9_1Display.SelectOrientationVectors = 'None'
a9_1Display.ScaleFactor = 0.00297174621373415
a9_1Display.SelectScaleArray = 'Eps_v'
a9_1Display.GlyphType = 'Arrow'
a9_1Display.GlyphTableIndexArray = 'Eps'
a9_1Display.GaussianRadius = 0.0001485873106867075
a9_1Display.SetScaleArray = ['POINTS', 'Eps']
a9_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a9_1Display.OpacityArray = ['POINTS', 'Eps']
a9_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a9_1Display.DataAxesGrid = 'GridAxesRepresentation'
a9_1Display.PolarAxes = 'PolarAxesRepresentation'
a9_1Display.ScalarOpacityFunction = pk1PWF
a9_1Display.ScalarOpacityUnitDistance = 0.0026211348082391347
a9_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a9_1Display.ScaleTransferFunction.Points = [0.0211469829082489, 0.0, 0.5, 0.0, 0.7001282109154595, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a9_1Display.OpacityTransferFunction.Points = [0.0211469829082489, 0.0, 0.5, 0.0, 0.7001282109154595, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1LUT in view renderView19
pk1LUTColorBar_4 = GetScalarBar(pk1LUT, renderView19)
pk1LUTColorBar_4.Title = 'pk1'
pk1LUTColorBar_4.ComponentTitle = '0'

# set color bar visibility
pk1LUTColorBar_4.Visibility = 1

# show color legend
a9_1Display.SetScalarBarVisibility(renderView19, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from a0_1
a0_1Display_1 = Show(a0_1, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a0_1Display_1.Representation = 'Surface'
a0_1Display_1.ColorArrayName = ['POINTS', 'health']
a0_1Display_1.LookupTable = healthLUT
a0_1Display_1.SelectTCoordArray = 'None'
a0_1Display_1.SelectNormalArray = 'None'
a0_1Display_1.SelectTangentArray = 'None'
a0_1Display_1.OSPRayScaleArray = 'Eps'
a0_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a0_1Display_1.SelectOrientationVectors = 'None'
a0_1Display_1.ScaleFactor = 0.002915369160473347
a0_1Display_1.SelectScaleArray = 'Eps_v'
a0_1Display_1.GlyphType = 'Arrow'
a0_1Display_1.GlyphTableIndexArray = 'Eps'
a0_1Display_1.GaussianRadius = 0.00014576845802366734
a0_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a0_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a0_1Display_1.OpacityArray = ['POINTS', 'Eps']
a0_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a0_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a0_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a0_1Display_1.ScalarOpacityFunction = healthPWF
a0_1Display_1.ScalarOpacityUnitDistance = 0.0025871796044513147
a0_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a0_1Display_1.ScaleTransferFunction.Points = [0.021653294563293457, 0.0, 0.5, 0.0, 0.5954719483852386, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a0_1Display_1.OpacityTransferFunction.Points = [0.021653294563293457, 0.0, 0.5, 0.0, 0.5954719483852386, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView2
healthLUTColorBar_5 = GetScalarBar(healthLUT, renderView2)
healthLUTColorBar_5.WindowLocation = 'Any Location'
healthLUTColorBar_5.Position = [0.6870642912470953, 0.2472527472527472]
healthLUTColorBar_5.Title = 'Health'
healthLUTColorBar_5.ComponentTitle = ''
healthLUTColorBar_5.TitleColor = [0.0, 0.0, 0.0]
healthLUTColorBar_5.TitleFontSize = 25
healthLUTColorBar_5.LabelColor = [0.0, 0.0, 0.0]
healthLUTColorBar_5.LabelFontSize = 20
healthLUTColorBar_5.ScalarBarLength = 0.5442857142857145

# set color bar visibility
healthLUTColorBar_5.Visibility = 1

# show color legend
a0_1Display_1.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView20'
# ----------------------------------------------------------------

# show data from a9_1
a9_1Display_1 = Show(a9_1, renderView20, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a9_1Display_1.Representation = 'Surface'
a9_1Display_1.ColorArrayName = ['POINTS', 'health']
a9_1Display_1.LookupTable = healthLUT
a9_1Display_1.SelectTCoordArray = 'None'
a9_1Display_1.SelectNormalArray = 'None'
a9_1Display_1.SelectTangentArray = 'None'
a9_1Display_1.OSPRayScaleArray = 'Eps'
a9_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a9_1Display_1.SelectOrientationVectors = 'None'
a9_1Display_1.ScaleFactor = 0.00297174621373415
a9_1Display_1.SelectScaleArray = 'Eps_v'
a9_1Display_1.GlyphType = 'Arrow'
a9_1Display_1.GlyphTableIndexArray = 'Eps'
a9_1Display_1.GaussianRadius = 0.0001485873106867075
a9_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a9_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a9_1Display_1.OpacityArray = ['POINTS', 'Eps']
a9_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a9_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a9_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a9_1Display_1.ScalarOpacityFunction = healthPWF
a9_1Display_1.ScalarOpacityUnitDistance = 0.0026211348082391347
a9_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a9_1Display_1.ScaleTransferFunction.Points = [0.0211469829082489, 0.0, 0.5, 0.0, 0.7001282109154595, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a9_1Display_1.OpacityTransferFunction.Points = [0.0211469829082489, 0.0, 0.5, 0.0, 0.7001282109154595, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView20
healthLUTColorBar_6 = GetScalarBar(healthLUT, renderView20)
healthLUTColorBar_6.Title = 'health'
healthLUTColorBar_6.ComponentTitle = ''

# set color bar visibility
healthLUTColorBar_6.Visibility = 1

# show color legend
a9_1Display_1.SetScalarBarVisibility(renderView20, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView21'
# ----------------------------------------------------------------

# show data from a10_1
a10_1Display = Show(a10_1, renderView21, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a10_1Display.Representation = 'Surface'
a10_1Display.ColorArrayName = ['POINTS', 'pk1']
a10_1Display.LookupTable = pk1LUT
a10_1Display.SelectTCoordArray = 'None'
a10_1Display.SelectNormalArray = 'None'
a10_1Display.SelectTangentArray = 'None'
a10_1Display.OSPRayScaleArray = 'Eps'
a10_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a10_1Display.SelectOrientationVectors = 'None'
a10_1Display.ScaleFactor = 0.003055161237716675
a10_1Display.SelectScaleArray = 'Eps_v'
a10_1Display.GlyphType = 'Arrow'
a10_1Display.GlyphTableIndexArray = 'Eps'
a10_1Display.GaussianRadius = 0.00015275806188583374
a10_1Display.SetScaleArray = ['POINTS', 'Eps']
a10_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a10_1Display.OpacityArray = ['POINTS', 'Eps']
a10_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a10_1Display.DataAxesGrid = 'GridAxesRepresentation'
a10_1Display.PolarAxes = 'PolarAxesRepresentation'
a10_1Display.ScalarOpacityFunction = pk1PWF
a10_1Display.ScalarOpacityUnitDistance = 0.0026717458544475623
a10_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a10_1Display.ScaleTransferFunction.Points = [0.018364280462265015, 0.0, 0.5, 0.0, 0.7617168596812657, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a10_1Display.OpacityTransferFunction.Points = [0.018364280462265015, 0.0, 0.5, 0.0, 0.7617168596812657, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1LUT in view renderView21
pk1LUTColorBar_5 = GetScalarBar(pk1LUT, renderView21)
pk1LUTColorBar_5.Title = 'pk1'
pk1LUTColorBar_5.ComponentTitle = '0'

# set color bar visibility
pk1LUTColorBar_5.Visibility = 1

# show color legend
a10_1Display.SetScalarBarVisibility(renderView21, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView22'
# ----------------------------------------------------------------

# show data from a10_1
a10_1Display_1 = Show(a10_1, renderView22, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a10_1Display_1.Representation = 'Surface'
a10_1Display_1.ColorArrayName = ['POINTS', 'health']
a10_1Display_1.LookupTable = healthLUT
a10_1Display_1.SelectTCoordArray = 'None'
a10_1Display_1.SelectNormalArray = 'None'
a10_1Display_1.SelectTangentArray = 'None'
a10_1Display_1.OSPRayScaleArray = 'Eps'
a10_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a10_1Display_1.SelectOrientationVectors = 'None'
a10_1Display_1.ScaleFactor = 0.003055161237716675
a10_1Display_1.SelectScaleArray = 'Eps_v'
a10_1Display_1.GlyphType = 'Arrow'
a10_1Display_1.GlyphTableIndexArray = 'Eps'
a10_1Display_1.GaussianRadius = 0.00015275806188583374
a10_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a10_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a10_1Display_1.OpacityArray = ['POINTS', 'Eps']
a10_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a10_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a10_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a10_1Display_1.ScalarOpacityFunction = healthPWF
a10_1Display_1.ScalarOpacityUnitDistance = 0.0026717458544475623
a10_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a10_1Display_1.ScaleTransferFunction.Points = [0.018364280462265015, 0.0, 0.5, 0.0, 0.7617168596812657, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a10_1Display_1.OpacityTransferFunction.Points = [0.018364280462265015, 0.0, 0.5, 0.0, 0.7617168596812657, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView22
healthLUTColorBar_7 = GetScalarBar(healthLUT, renderView22)
healthLUTColorBar_7.Title = 'health'
healthLUTColorBar_7.ComponentTitle = ''

# set color bar visibility
healthLUTColorBar_7.Visibility = 1

# show color legend
a10_1Display_1.SetScalarBarVisibility(renderView22, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView23'
# ----------------------------------------------------------------

# show data from net0
net0Display = Show(net0, renderView23, 'StructuredGridRepresentation')

# get color transfer function/color map for 'pk1_11'
pk1_11LUT = GetColorTransferFunction('pk1_11')
pk1_11LUT.AutomaticRescaleRangeMode = 'Never'
pk1_11LUT.RGBPoints = [15000.0, 0.02, 0.3813, 0.9981, 15833.333333333334, 0.02000006, 0.424267768, 0.96906969, 16666.666666666668, 0.02, 0.467233763, 0.940033043, 17500.0, 0.02, 0.5102, 0.911, 18333.333333333332, 0.02000006, 0.546401494, 0.872669438, 19166.666666666664, 0.02, 0.582600362, 0.83433295, 20000.0, 0.02, 0.6188, 0.796, 20833.333333333332, 0.02000006, 0.652535156, 0.749802434, 21666.666666666664, 0.02, 0.686267004, 0.703599538, 22500.0, 0.02, 0.72, 0.6574, 23333.333333333332, 0.02000006, 0.757035456, 0.603735359, 24166.666666666668, 0.02, 0.794067037, 0.55006613, 25000.0, 0.02, 0.8311, 0.4964, 25833.333333333336, 0.021354336738172372, 0.8645368555261631, 0.4285579460761159, 26666.666666666664, 0.023312914349117714, 0.897999359924484, 0.36073871343115577, 27500.0, 0.015976108242848862, 0.9310479513349017, 0.2925631815088092, 28333.333333333332, 0.27421074700988196, 0.952562960995083, 0.15356836602739213, 29166.666666666664, 0.4933546281681699, 0.9619038625309482, 0.11119493614749336, 30000.0, 0.6439, 0.9773, 0.0469, 30833.333333333336, 0.762401813, 0.984669591, 0.034600153, 31666.666666666664, 0.880901185, 0.992033407, 0.022299877, 32500.0, 0.9995285432627147, 0.9995193706781492, 0.0134884641450013, 33333.333333333336, 0.999402998, 0.955036376, 0.079066628, 34166.66666666667, 0.9994, 0.910666223, 0.148134024, 35000.0, 0.9994, 0.8663, 0.2172, 35833.33333333333, 0.999269665, 0.818035981, 0.217200652, 36666.66666666667, 0.999133332, 0.769766184, 0.2172, 37500.0, 0.999, 0.7215, 0.2172, 38333.33333333333, 0.99913633, 0.673435546, 0.217200652, 39166.66666666667, 0.999266668, 0.625366186, 0.2172, 40000.0, 0.9994, 0.5773, 0.2172, 40833.333333333336, 0.999402998, 0.521068455, 0.217200652, 41666.666666666664, 0.9994, 0.464832771, 0.2172, 42500.0, 0.9994, 0.4086, 0.2172, 43333.33333333333, 0.9947599917687346, 0.33177297300202935, 0.2112309638520206, 44166.66666666667, 0.9867129505479589, 0.2595183410914934, 0.19012239549291934, 45000.0, 0.9912458875646419, 0.14799417507952672, 0.21078892136920357, 45833.33333333333, 0.949903037, 0.116867171, 0.252900603, 46666.66666666667, 0.903199533, 0.078432949, 0.291800389, 47500.0, 0.8565, 0.04, 0.3307, 48333.33333333333, 0.798902627, 0.04333345, 0.358434298, 49166.666666666664, 0.741299424, 0.0466667, 0.386166944, 50000.0, 0.6837, 0.05, 0.4139]
pk1_11LUT.ColorSpace = 'RGB'
pk1_11LUT.NanColor = [1.0, 0.0, 0.0]
pk1_11LUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'pk1_11'
pk1_11PWF = GetOpacityTransferFunction('pk1_11')
pk1_11PWF.Points = [15000.0, 0.0, 0.5, 0.0, 50000.0, 1.0, 0.5, 0.0]
pk1_11PWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
net0Display.Representation = 'Surface'
net0Display.ColorArrayName = ['POINTS', 'pk1_11']
net0Display.LookupTable = pk1_11LUT
net0Display.SelectTCoordArray = 'None'
net0Display.SelectNormalArray = 'None'
net0Display.SelectTangentArray = 'None'
net0Display.EdgeColor = [0.9647058823529412, 0.9607843137254902, 0.9568627450980393]
net0Display.OSPRayScaleArray = 'S-VonMises'
net0Display.OSPRayScaleFunction = 'PiecewiseFunction'
net0Display.SelectOrientationVectors = 'displacement'
net0Display.ScaleFactor = 0.29918588399887086
net0Display.SelectScaleArray = 'S-VonMises'
net0Display.GlyphType = 'Arrow'
net0Display.GlyphTableIndexArray = 'S-VonMises'
net0Display.GaussianRadius = 0.014959294199943542
net0Display.SetScaleArray = ['POINTS', 'S-VonMises']
net0Display.ScaleTransferFunction = 'PiecewiseFunction'
net0Display.OpacityArray = ['POINTS', 'S-VonMises']
net0Display.OpacityTransferFunction = 'PiecewiseFunction'
net0Display.DataAxesGrid = 'GridAxesRepresentation'
net0Display.PolarAxes = 'PolarAxesRepresentation'
net0Display.ScalarOpacityFunction = pk1_11PWF
net0Display.ScalarOpacityUnitDistance = 0.3006211084111547

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net0Display.ScaleTransferFunction.Points = [4503.316306688898, 0.0, 0.5, 0.0, 72956.586662639, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net0Display.OpacityTransferFunction.Points = [4503.316306688898, 0.0, 0.5, 0.0, 72956.586662639, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView23
pk1_11LUTColorBar = GetScalarBar(pk1_11LUT, renderView23)
pk1_11LUTColorBar.WindowLocation = 'Any Location'
pk1_11LUTColorBar.Position = [0.6844500387296667, 0.20348837209302328]
pk1_11LUTColorBar.Title = '$P_{xx}$'
pk1_11LUTColorBar.ComponentTitle = ''
pk1_11LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
pk1_11LUTColorBar.TitleFontSize = 25
pk1_11LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
pk1_11LUTColorBar.LabelFontSize = 20
pk1_11LUTColorBar.ScalarBarLength = 0.574186046511628

# set color bar visibility
pk1_11LUTColorBar.Visibility = 1

# show color legend
net0Display.SetScalarBarVisibility(renderView23, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView24'
# ----------------------------------------------------------------

# show data from net_1
net_1Display = Show(net_1, renderView24, 'StructuredGridRepresentation')

# trace defaults for the display properties.
net_1Display.Representation = 'Surface'
net_1Display.ColorArrayName = ['POINTS', 'pk1_11']
net_1Display.LookupTable = pk1_11LUT
net_1Display.SelectTCoordArray = 'None'
net_1Display.SelectNormalArray = 'None'
net_1Display.SelectTangentArray = 'None'
net_1Display.OSPRayScaleArray = 'S-VonMises'
net_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
net_1Display.SelectOrientationVectors = 'displacement'
net_1Display.ScaleFactor = 0.3008497714996338
net_1Display.SelectScaleArray = 'S-VonMises'
net_1Display.GlyphType = 'Arrow'
net_1Display.GlyphTableIndexArray = 'S-VonMises'
net_1Display.GaussianRadius = 0.01504248857498169
net_1Display.SetScaleArray = ['POINTS', 'S-VonMises']
net_1Display.ScaleTransferFunction = 'PiecewiseFunction'
net_1Display.OpacityArray = ['POINTS', 'S-VonMises']
net_1Display.OpacityTransferFunction = 'PiecewiseFunction'
net_1Display.DataAxesGrid = 'GridAxesRepresentation'
net_1Display.PolarAxes = 'PolarAxesRepresentation'
net_1Display.ScalarOpacityFunction = pk1_11PWF
net_1Display.ScalarOpacityUnitDistance = 0.30177206378345606

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net_1Display.ScaleTransferFunction.Points = [3178.229328083513, 0.0, 0.5, 0.0, 87920.0476473885, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net_1Display.OpacityTransferFunction.Points = [3178.229328083513, 0.0, 0.5, 0.0, 87920.0476473885, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView24
pk1_11LUTColorBar_1 = GetScalarBar(pk1_11LUT, renderView24)
pk1_11LUTColorBar_1.Title = 'pk1_11'
pk1_11LUTColorBar_1.ComponentTitle = ''

# set color bar visibility
pk1_11LUTColorBar_1.Visibility = 1

# show color legend
net_1Display.SetScalarBarVisibility(renderView24, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView25'
# ----------------------------------------------------------------

# show data from net_3
net_3Display = Show(net_3, renderView25, 'StructuredGridRepresentation')

# trace defaults for the display properties.
net_3Display.Representation = 'Surface'
net_3Display.ColorArrayName = ['POINTS', 'pk1_11']
net_3Display.LookupTable = pk1_11LUT
net_3Display.SelectTCoordArray = 'None'
net_3Display.SelectNormalArray = 'None'
net_3Display.SelectTangentArray = 'None'
net_3Display.OSPRayScaleArray = 'S-VonMises'
net_3Display.OSPRayScaleFunction = 'PiecewiseFunction'
net_3Display.SelectOrientationVectors = 'displacement'
net_3Display.ScaleFactor = 0.29670700430870056
net_3Display.SelectScaleArray = 'S-VonMises'
net_3Display.GlyphType = 'Arrow'
net_3Display.GlyphTableIndexArray = 'S-VonMises'
net_3Display.GaussianRadius = 0.014835350215435028
net_3Display.SetScaleArray = ['POINTS', 'S-VonMises']
net_3Display.ScaleTransferFunction = 'PiecewiseFunction'
net_3Display.OpacityArray = ['POINTS', 'S-VonMises']
net_3Display.OpacityTransferFunction = 'PiecewiseFunction'
net_3Display.DataAxesGrid = 'GridAxesRepresentation'
net_3Display.PolarAxes = 'PolarAxesRepresentation'
net_3Display.ScalarOpacityFunction = pk1_11PWF
net_3Display.ScalarOpacityUnitDistance = 0.2989101109840586

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net_3Display.ScaleTransferFunction.Points = [2582.267193054545, 0.0, 0.5, 0.0, 66731.47100155057, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net_3Display.OpacityTransferFunction.Points = [2582.267193054545, 0.0, 0.5, 0.0, 66731.47100155057, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView25
pk1_11LUTColorBar_2 = GetScalarBar(pk1_11LUT, renderView25)
pk1_11LUTColorBar_2.Title = 'pk1_11'
pk1_11LUTColorBar_2.ComponentTitle = ''

# set color bar visibility
pk1_11LUTColorBar_2.Visibility = 1

# show color legend
net_3Display.SetScalarBarVisibility(renderView25, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView26'
# ----------------------------------------------------------------

# show data from net_4
net_4Display = Show(net_4, renderView26, 'StructuredGridRepresentation')

# trace defaults for the display properties.
net_4Display.Representation = 'Surface'
net_4Display.ColorArrayName = ['POINTS', 'pk1_11']
net_4Display.LookupTable = pk1_11LUT
net_4Display.SelectTCoordArray = 'None'
net_4Display.SelectNormalArray = 'None'
net_4Display.SelectTangentArray = 'None'
net_4Display.OSPRayScaleArray = 'S-VonMises'
net_4Display.OSPRayScaleFunction = 'PiecewiseFunction'
net_4Display.SelectOrientationVectors = 'displacement'
net_4Display.ScaleFactor = 0.30745333433151245
net_4Display.SelectScaleArray = 'S-VonMises'
net_4Display.GlyphType = 'Arrow'
net_4Display.GlyphTableIndexArray = 'S-VonMises'
net_4Display.GaussianRadius = 0.015372666716575624
net_4Display.SetScaleArray = ['POINTS', 'S-VonMises']
net_4Display.ScaleTransferFunction = 'PiecewiseFunction'
net_4Display.OpacityArray = ['POINTS', 'S-VonMises']
net_4Display.OpacityTransferFunction = 'PiecewiseFunction'
net_4Display.DataAxesGrid = 'GridAxesRepresentation'
net_4Display.PolarAxes = 'PolarAxesRepresentation'
net_4Display.ScalarOpacityFunction = pk1_11PWF
net_4Display.ScalarOpacityUnitDistance = 0.3063591673813379

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net_4Display.ScaleTransferFunction.Points = [4358.420784099814, 0.0, 0.5, 0.0, 107998.4968477927, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net_4Display.OpacityTransferFunction.Points = [4358.420784099814, 0.0, 0.5, 0.0, 107998.4968477927, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView26
pk1_11LUTColorBar_3 = GetScalarBar(pk1_11LUT, renderView26)
pk1_11LUTColorBar_3.Title = 'pk1_11'
pk1_11LUTColorBar_3.ComponentTitle = ''

# set color bar visibility
pk1_11LUTColorBar_3.Visibility = 1

# show color legend
net_4Display.SetScalarBarVisibility(renderView26, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView27'
# ----------------------------------------------------------------

# show data from net_5
net_5Display = Show(net_5, renderView27, 'StructuredGridRepresentation')

# trace defaults for the display properties.
net_5Display.Representation = 'Surface'
net_5Display.ColorArrayName = ['POINTS', 'pk1_11']
net_5Display.LookupTable = pk1_11LUT
net_5Display.SelectTCoordArray = 'None'
net_5Display.SelectNormalArray = 'None'
net_5Display.SelectTangentArray = 'None'
net_5Display.OSPRayScaleArray = 'S-VonMises'
net_5Display.OSPRayScaleFunction = 'PiecewiseFunction'
net_5Display.SelectOrientationVectors = 'displacement'
net_5Display.ScaleFactor = 0.305462908744812
net_5Display.SelectScaleArray = 'S-VonMises'
net_5Display.GlyphType = 'Arrow'
net_5Display.GlyphTableIndexArray = 'S-VonMises'
net_5Display.GaussianRadius = 0.015273145437240601
net_5Display.SetScaleArray = ['POINTS', 'S-VonMises']
net_5Display.ScaleTransferFunction = 'PiecewiseFunction'
net_5Display.OpacityArray = ['POINTS', 'S-VonMises']
net_5Display.OpacityTransferFunction = 'PiecewiseFunction'
net_5Display.DataAxesGrid = 'GridAxesRepresentation'
net_5Display.PolarAxes = 'PolarAxesRepresentation'
net_5Display.ScalarOpacityFunction = pk1_11PWF
net_5Display.ScalarOpacityUnitDistance = 0.3049733465344432

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net_5Display.ScaleTransferFunction.Points = [3533.7784654126544, 0.0, 0.5, 0.0, 107007.39923125286, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net_5Display.OpacityTransferFunction.Points = [3533.7784654126544, 0.0, 0.5, 0.0, 107007.39923125286, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView27
pk1_11LUTColorBar_4 = GetScalarBar(pk1_11LUT, renderView27)
pk1_11LUTColorBar_4.Title = 'pk1_11'
pk1_11LUTColorBar_4.ComponentTitle = ''

# set color bar visibility
pk1_11LUTColorBar_4.Visibility = 1

# show color legend
net_5Display.SetScalarBarVisibility(renderView27, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView28'
# ----------------------------------------------------------------

# show data from net_6
net_6Display = Show(net_6, renderView28, 'StructuredGridRepresentation')

# trace defaults for the display properties.
net_6Display.Representation = 'Surface'
net_6Display.ColorArrayName = ['POINTS', 'pk1_11']
net_6Display.LookupTable = pk1_11LUT
net_6Display.SelectTCoordArray = 'None'
net_6Display.SelectNormalArray = 'None'
net_6Display.SelectTangentArray = 'None'
net_6Display.OSPRayScaleArray = 'S-VonMises'
net_6Display.OSPRayScaleFunction = 'PiecewiseFunction'
net_6Display.SelectOrientationVectors = 'displacement'
net_6Display.ScaleFactor = 0.30478012561798096
net_6Display.SelectScaleArray = 'S-VonMises'
net_6Display.GlyphType = 'Arrow'
net_6Display.GlyphTableIndexArray = 'S-VonMises'
net_6Display.GaussianRadius = 0.015239006280899048
net_6Display.SetScaleArray = ['POINTS', 'S-VonMises']
net_6Display.ScaleTransferFunction = 'PiecewiseFunction'
net_6Display.OpacityArray = ['POINTS', 'S-VonMises']
net_6Display.OpacityTransferFunction = 'PiecewiseFunction'
net_6Display.DataAxesGrid = 'GridAxesRepresentation'
net_6Display.PolarAxes = 'PolarAxesRepresentation'
net_6Display.ScalarOpacityFunction = pk1_11PWF
net_6Display.ScalarOpacityUnitDistance = 0.30449859169044907

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net_6Display.ScaleTransferFunction.Points = [6166.581859536554, 0.0, 0.5, 0.0, 78387.60776990121, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net_6Display.OpacityTransferFunction.Points = [6166.581859536554, 0.0, 0.5, 0.0, 78387.60776990121, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView28
pk1_11LUTColorBar_5 = GetScalarBar(pk1_11LUT, renderView28)
pk1_11LUTColorBar_5.Title = 'pk1_11'
pk1_11LUTColorBar_5.ComponentTitle = ''

# set color bar visibility
pk1_11LUTColorBar_5.Visibility = 1

# show color legend
net_6Display.SetScalarBarVisibility(renderView28, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView29'
# ----------------------------------------------------------------

# show data from net_2
net_2Display = Show(net_2, renderView29, 'StructuredGridRepresentation')

# trace defaults for the display properties.
net_2Display.Representation = 'Surface'
net_2Display.ColorArrayName = ['POINTS', 'pk1_11']
net_2Display.LookupTable = pk1_11LUT
net_2Display.SelectTCoordArray = 'None'
net_2Display.SelectNormalArray = 'None'
net_2Display.SelectTangentArray = 'None'
net_2Display.OSPRayScaleArray = 'S-VonMises'
net_2Display.OSPRayScaleFunction = 'PiecewiseFunction'
net_2Display.SelectOrientationVectors = 'displacement'
net_2Display.ScaleFactor = 0.29997475147247316
net_2Display.SelectScaleArray = 'S-VonMises'
net_2Display.GlyphType = 'Arrow'
net_2Display.GlyphTableIndexArray = 'S-VonMises'
net_2Display.GaussianRadius = 0.014998737573623657
net_2Display.SetScaleArray = ['POINTS', 'S-VonMises']
net_2Display.ScaleTransferFunction = 'PiecewiseFunction'
net_2Display.OpacityArray = ['POINTS', 'S-VonMises']
net_2Display.OpacityTransferFunction = 'PiecewiseFunction'
net_2Display.DataAxesGrid = 'GridAxesRepresentation'
net_2Display.PolarAxes = 'PolarAxesRepresentation'
net_2Display.ScalarOpacityFunction = pk1_11PWF
net_2Display.ScalarOpacityUnitDistance = 0.3011665415985859

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net_2Display.ScaleTransferFunction.Points = [4116.051885504925, 0.0, 0.5, 0.0, 74483.5481592342, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net_2Display.OpacityTransferFunction.Points = [4116.051885504925, 0.0, 0.5, 0.0, 74483.5481592342, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView29
pk1_11LUTColorBar_6 = GetScalarBar(pk1_11LUT, renderView29)
pk1_11LUTColorBar_6.Title = 'pk1_11'
pk1_11LUTColorBar_6.ComponentTitle = ''

# set color bar visibility
pk1_11LUTColorBar_6.Visibility = 1

# show color legend
net_2Display.SetScalarBarVisibility(renderView29, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView3'
# ----------------------------------------------------------------

# show data from a1_1
a1_1Display = Show(a1_1, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a1_1Display.Representation = 'Surface'
a1_1Display.ColorArrayName = ['POINTS', 'pk1']
a1_1Display.LookupTable = pk1LUT
a1_1Display.SelectTCoordArray = 'None'
a1_1Display.SelectNormalArray = 'None'
a1_1Display.SelectTangentArray = 'None'
a1_1Display.OSPRayScaleArray = 'Eps'
a1_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a1_1Display.SelectOrientationVectors = 'None'
a1_1Display.ScaleFactor = 0.002910099364817143
a1_1Display.SelectScaleArray = 'Eps_v'
a1_1Display.GlyphType = 'Arrow'
a1_1Display.GlyphTableIndexArray = 'Eps'
a1_1Display.GaussianRadius = 0.00014550496824085711
a1_1Display.SetScaleArray = ['POINTS', 'Eps']
a1_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a1_1Display.OpacityArray = ['POINTS', 'Eps']
a1_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a1_1Display.DataAxesGrid = 'GridAxesRepresentation'
a1_1Display.PolarAxes = 'PolarAxesRepresentation'
a1_1Display.ScalarOpacityFunction = pk1PWF
a1_1Display.ScalarOpacityUnitDistance = 0.002584016363643278
a1_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a1_1Display.ScaleTransferFunction.Points = [0.01901465654373169, 0.0, 0.5, 0.0, 0.7123012031827654, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a1_1Display.OpacityTransferFunction.Points = [0.01901465654373169, 0.0, 0.5, 0.0, 0.7123012031827654, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1LUT in view renderView3
pk1LUTColorBar_6 = GetScalarBar(pk1LUT, renderView3)
pk1LUTColorBar_6.Title = 'pk1'
pk1LUTColorBar_6.ComponentTitle = '0'

# set color bar visibility
pk1LUTColorBar_6.Visibility = 1

# show color legend
a1_1Display.SetScalarBarVisibility(renderView3, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView30'
# ----------------------------------------------------------------

# show data from net_7
net_7Display = Show(net_7, renderView30, 'StructuredGridRepresentation')

# trace defaults for the display properties.
net_7Display.Representation = 'Surface'
net_7Display.ColorArrayName = ['POINTS', 'pk1_11']
net_7Display.LookupTable = pk1_11LUT
net_7Display.SelectTCoordArray = 'None'
net_7Display.SelectNormalArray = 'None'
net_7Display.SelectTangentArray = 'None'
net_7Display.OSPRayScaleArray = 'S-VonMises'
net_7Display.OSPRayScaleFunction = 'PiecewiseFunction'
net_7Display.SelectOrientationVectors = 'displacement'
net_7Display.ScaleFactor = 0.30267834663391113
net_7Display.SelectScaleArray = 'S-VonMises'
net_7Display.GlyphType = 'Arrow'
net_7Display.GlyphTableIndexArray = 'S-VonMises'
net_7Display.GaussianRadius = 0.015133917331695557
net_7Display.SetScaleArray = ['POINTS', 'S-VonMises']
net_7Display.ScaleTransferFunction = 'PiecewiseFunction'
net_7Display.OpacityArray = ['POINTS', 'S-VonMises']
net_7Display.OpacityTransferFunction = 'PiecewiseFunction'
net_7Display.DataAxesGrid = 'GridAxesRepresentation'
net_7Display.PolarAxes = 'PolarAxesRepresentation'
net_7Display.ScalarOpacityFunction = pk1_11PWF
net_7Display.ScalarOpacityUnitDistance = 0.3030392135436688

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net_7Display.ScaleTransferFunction.Points = [5792.159222380092, 0.0, 0.5, 0.0, 77329.53600623061, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net_7Display.OpacityTransferFunction.Points = [5792.159222380092, 0.0, 0.5, 0.0, 77329.53600623061, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView30
pk1_11LUTColorBar_7 = GetScalarBar(pk1_11LUT, renderView30)
pk1_11LUTColorBar_7.Title = 'pk1_11'
pk1_11LUTColorBar_7.ComponentTitle = ''

# set color bar visibility
pk1_11LUTColorBar_7.Visibility = 1

# show color legend
net_7Display.SetScalarBarVisibility(renderView30, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView31'
# ----------------------------------------------------------------

# show data from net_8
net_8Display = Show(net_8, renderView31, 'StructuredGridRepresentation')

# trace defaults for the display properties.
net_8Display.Representation = 'Surface'
net_8Display.ColorArrayName = ['POINTS', 'pk1_11']
net_8Display.LookupTable = pk1_11LUT
net_8Display.SelectTCoordArray = 'None'
net_8Display.SelectNormalArray = 'None'
net_8Display.SelectTangentArray = 'None'
net_8Display.OSPRayScaleArray = 'S-VonMises'
net_8Display.OSPRayScaleFunction = 'PiecewiseFunction'
net_8Display.SelectOrientationVectors = 'displacement'
net_8Display.ScaleFactor = 0.2973787546157837
net_8Display.SelectScaleArray = 'S-VonMises'
net_8Display.GlyphType = 'Arrow'
net_8Display.GlyphTableIndexArray = 'S-VonMises'
net_8Display.GaussianRadius = 0.014868937730789185
net_8Display.SetScaleArray = ['POINTS', 'S-VonMises']
net_8Display.ScaleTransferFunction = 'PiecewiseFunction'
net_8Display.OpacityArray = ['POINTS', 'S-VonMises']
net_8Display.OpacityTransferFunction = 'PiecewiseFunction'
net_8Display.DataAxesGrid = 'GridAxesRepresentation'
net_8Display.PolarAxes = 'PolarAxesRepresentation'
net_8Display.ScalarOpacityFunction = pk1_11PWF
net_8Display.ScalarOpacityUnitDistance = 0.29937333123732823

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net_8Display.ScaleTransferFunction.Points = [4731.908874665344, 0.0, 0.5, 0.0, 81357.96623880052, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net_8Display.OpacityTransferFunction.Points = [4731.908874665344, 0.0, 0.5, 0.0, 81357.96623880052, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView31
pk1_11LUTColorBar_8 = GetScalarBar(pk1_11LUT, renderView31)
pk1_11LUTColorBar_8.Title = 'pk1_11'
pk1_11LUTColorBar_8.ComponentTitle = ''

# set color bar visibility
pk1_11LUTColorBar_8.Visibility = 1

# show color legend
net_8Display.SetScalarBarVisibility(renderView31, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView32'
# ----------------------------------------------------------------

# show data from net_9
net_9Display = Show(net_9, renderView32, 'StructuredGridRepresentation')

# trace defaults for the display properties.
net_9Display.Representation = 'Surface'
net_9Display.ColorArrayName = ['POINTS', 'pk1_11']
net_9Display.LookupTable = pk1_11LUT
net_9Display.SelectTCoordArray = 'None'
net_9Display.SelectNormalArray = 'None'
net_9Display.SelectTangentArray = 'None'
net_9Display.OSPRayScaleArray = 'S-VonMises'
net_9Display.OSPRayScaleFunction = 'PiecewiseFunction'
net_9Display.SelectOrientationVectors = 'displacement'
net_9Display.ScaleFactor = 0.30712696313858034
net_9Display.SelectScaleArray = 'S-VonMises'
net_9Display.GlyphType = 'Arrow'
net_9Display.GlyphTableIndexArray = 'S-VonMises'
net_9Display.GaussianRadius = 0.015356348156929017
net_9Display.SetScaleArray = ['POINTS', 'S-VonMises']
net_9Display.ScaleTransferFunction = 'PiecewiseFunction'
net_9Display.OpacityArray = ['POINTS', 'S-VonMises']
net_9Display.OpacityTransferFunction = 'PiecewiseFunction'
net_9Display.DataAxesGrid = 'GridAxesRepresentation'
net_9Display.PolarAxes = 'PolarAxesRepresentation'
net_9Display.ScalarOpacityFunction = pk1_11PWF
net_9Display.ScalarOpacityUnitDistance = 0.3061317475626834

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net_9Display.ScaleTransferFunction.Points = [4765.1199946150755, 0.0, 0.5, 0.0, 87150.8718733395, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net_9Display.OpacityTransferFunction.Points = [4765.1199946150755, 0.0, 0.5, 0.0, 87150.8718733395, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView32
pk1_11LUTColorBar_9 = GetScalarBar(pk1_11LUT, renderView32)
pk1_11LUTColorBar_9.Title = 'pk1_11'
pk1_11LUTColorBar_9.ComponentTitle = ''

# set color bar visibility
pk1_11LUTColorBar_9.Visibility = 1

# show color legend
net_9Display.SetScalarBarVisibility(renderView32, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView33'
# ----------------------------------------------------------------

# show data from net_10
net_10Display = Show(net_10, renderView33, 'StructuredGridRepresentation')

# trace defaults for the display properties.
net_10Display.Representation = 'Surface'
net_10Display.ColorArrayName = ['POINTS', 'pk1_11']
net_10Display.LookupTable = pk1_11LUT
net_10Display.SelectTCoordArray = 'None'
net_10Display.SelectNormalArray = 'None'
net_10Display.SelectTangentArray = 'None'
net_10Display.OSPRayScaleArray = 'S-VonMises'
net_10Display.OSPRayScaleFunction = 'PiecewiseFunction'
net_10Display.SelectOrientationVectors = 'displacement'
net_10Display.ScaleFactor = 0.3087336540222168
net_10Display.SelectScaleArray = 'S-VonMises'
net_10Display.GlyphType = 'Arrow'
net_10Display.GlyphTableIndexArray = 'S-VonMises'
net_10Display.GaussianRadius = 0.01543668270111084
net_10Display.SetScaleArray = ['POINTS', 'S-VonMises']
net_10Display.ScaleTransferFunction = 'PiecewiseFunction'
net_10Display.OpacityArray = ['POINTS', 'S-VonMises']
net_10Display.OpacityTransferFunction = 'PiecewiseFunction'
net_10Display.DataAxesGrid = 'GridAxesRepresentation'
net_10Display.PolarAxes = 'PolarAxesRepresentation'
net_10Display.ScalarOpacityFunction = pk1_11PWF
net_10Display.ScalarOpacityUnitDistance = 0.30725201057780577

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
net_10Display.ScaleTransferFunction.Points = [5692.759647766988, 0.0, 0.5, 0.0, 81600.72604284076, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
net_10Display.OpacityTransferFunction.Points = [5692.759647766988, 0.0, 0.5, 0.0, 81600.72604284076, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1_11LUT in view renderView33
pk1_11LUTColorBar_10 = GetScalarBar(pk1_11LUT, renderView33)
pk1_11LUTColorBar_10.Title = 'pk1_11'
pk1_11LUTColorBar_10.ComponentTitle = ''

# set color bar visibility
pk1_11LUTColorBar_10.Visibility = 1

# show color legend
net_10Display.SetScalarBarVisibility(renderView33, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView4'
# ----------------------------------------------------------------

# show data from a1_1
a1_1Display_1 = Show(a1_1, renderView4, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a1_1Display_1.Representation = 'Surface'
a1_1Display_1.ColorArrayName = ['POINTS', 'health']
a1_1Display_1.LookupTable = healthLUT
a1_1Display_1.SelectTCoordArray = 'None'
a1_1Display_1.SelectNormalArray = 'None'
a1_1Display_1.SelectTangentArray = 'None'
a1_1Display_1.OSPRayScaleArray = 'Eps'
a1_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a1_1Display_1.SelectOrientationVectors = 'None'
a1_1Display_1.ScaleFactor = 0.002910099364817143
a1_1Display_1.SelectScaleArray = 'Eps_v'
a1_1Display_1.GlyphType = 'Arrow'
a1_1Display_1.GlyphTableIndexArray = 'Eps'
a1_1Display_1.GaussianRadius = 0.00014550496824085711
a1_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a1_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a1_1Display_1.OpacityArray = ['POINTS', 'Eps']
a1_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a1_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a1_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a1_1Display_1.ScalarOpacityFunction = healthPWF
a1_1Display_1.ScalarOpacityUnitDistance = 0.002584016363643278
a1_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a1_1Display_1.ScaleTransferFunction.Points = [0.01901465654373169, 0.0, 0.5, 0.0, 0.7123012031827654, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a1_1Display_1.OpacityTransferFunction.Points = [0.01901465654373169, 0.0, 0.5, 0.0, 0.7123012031827654, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView4
healthLUTColorBar_8 = GetScalarBar(healthLUT, renderView4)
healthLUTColorBar_8.Title = 'health'
healthLUTColorBar_8.ComponentTitle = ''

# set color bar visibility
healthLUTColorBar_8.Visibility = 1

# show color legend
a1_1Display_1.SetScalarBarVisibility(renderView4, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView5'
# ----------------------------------------------------------------

# show data from a2_1
a2_1Display = Show(a2_1, renderView5, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a2_1Display.Representation = 'Surface'
a2_1Display.ColorArrayName = ['POINTS', 'pk1']
a2_1Display.LookupTable = pk1LUT
a2_1Display.SelectTCoordArray = 'None'
a2_1Display.SelectNormalArray = 'None'
a2_1Display.SelectTangentArray = 'None'
a2_1Display.OSPRayScaleArray = 'Eps'
a2_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a2_1Display.SelectOrientationVectors = 'None'
a2_1Display.ScaleFactor = 0.003024476766586304
a2_1Display.SelectScaleArray = 'Eps_v'
a2_1Display.GlyphType = 'Arrow'
a2_1Display.GlyphTableIndexArray = 'Eps'
a2_1Display.GaussianRadius = 0.0001512238383293152
a2_1Display.SetScaleArray = ['POINTS', 'Eps']
a2_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a2_1Display.OpacityArray = ['POINTS', 'Eps']
a2_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a2_1Display.DataAxesGrid = 'GridAxesRepresentation'
a2_1Display.PolarAxes = 'PolarAxesRepresentation'
a2_1Display.ScalarOpacityFunction = pk1PWF
a2_1Display.ScalarOpacityUnitDistance = 0.002653078201566743
a2_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a2_1Display.ScaleTransferFunction.Points = [0.021900564432144165, 0.0, 0.5, 0.0, 0.7783549070358277, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a2_1Display.OpacityTransferFunction.Points = [0.021900564432144165, 0.0, 0.5, 0.0, 0.7783549070358277, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1LUT in view renderView5
pk1LUTColorBar_7 = GetScalarBar(pk1LUT, renderView5)
pk1LUTColorBar_7.Title = 'pk1'
pk1LUTColorBar_7.ComponentTitle = '0'

# set color bar visibility
pk1LUTColorBar_7.Visibility = 1

# show color legend
a2_1Display.SetScalarBarVisibility(renderView5, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView6'
# ----------------------------------------------------------------

# show data from a2_1
a2_1Display_1 = Show(a2_1, renderView6, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a2_1Display_1.Representation = 'Surface'
a2_1Display_1.ColorArrayName = ['POINTS', 'health']
a2_1Display_1.LookupTable = healthLUT
a2_1Display_1.SelectTCoordArray = 'None'
a2_1Display_1.SelectNormalArray = 'None'
a2_1Display_1.SelectTangentArray = 'None'
a2_1Display_1.OSPRayScaleArray = 'Eps'
a2_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a2_1Display_1.SelectOrientationVectors = 'None'
a2_1Display_1.ScaleFactor = 0.003024476766586304
a2_1Display_1.SelectScaleArray = 'Eps_v'
a2_1Display_1.GlyphType = 'Arrow'
a2_1Display_1.GlyphTableIndexArray = 'Eps'
a2_1Display_1.GaussianRadius = 0.0001512238383293152
a2_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a2_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a2_1Display_1.OpacityArray = ['POINTS', 'Eps']
a2_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a2_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a2_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a2_1Display_1.ScalarOpacityFunction = healthPWF
a2_1Display_1.ScalarOpacityUnitDistance = 0.002653078201566743
a2_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a2_1Display_1.ScaleTransferFunction.Points = [0.021900564432144165, 0.0, 0.5, 0.0, 0.7783549070358277, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a2_1Display_1.OpacityTransferFunction.Points = [0.021900564432144165, 0.0, 0.5, 0.0, 0.7783549070358277, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView6
healthLUTColorBar_9 = GetScalarBar(healthLUT, renderView6)
healthLUTColorBar_9.Title = 'health'
healthLUTColorBar_9.ComponentTitle = ''

# set color bar visibility
healthLUTColorBar_9.Visibility = 1

# show color legend
a2_1Display_1.SetScalarBarVisibility(renderView6, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView7'
# ----------------------------------------------------------------

# show data from a3_1
a3_1Display = Show(a3_1, renderView7, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a3_1Display.Representation = 'Surface'
a3_1Display.ColorArrayName = ['POINTS', 'pk1']
a3_1Display.LookupTable = pk1LUT
a3_1Display.SelectTCoordArray = 'None'
a3_1Display.SelectNormalArray = 'None'
a3_1Display.SelectTangentArray = 'None'
a3_1Display.OSPRayScaleArray = 'Eps'
a3_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
a3_1Display.SelectOrientationVectors = 'None'
a3_1Display.ScaleFactor = 0.0029217138886451725
a3_1Display.SelectScaleArray = 'Eps_v'
a3_1Display.GlyphType = 'Arrow'
a3_1Display.GlyphTableIndexArray = 'Eps'
a3_1Display.GaussianRadius = 0.0001460856944322586
a3_1Display.SetScaleArray = ['POINTS', 'Eps']
a3_1Display.ScaleTransferFunction = 'PiecewiseFunction'
a3_1Display.OpacityArray = ['POINTS', 'Eps']
a3_1Display.OpacityTransferFunction = 'PiecewiseFunction'
a3_1Display.DataAxesGrid = 'GridAxesRepresentation'
a3_1Display.PolarAxes = 'PolarAxesRepresentation'
a3_1Display.ScalarOpacityFunction = pk1PWF
a3_1Display.ScalarOpacityUnitDistance = 0.002590990536386772
a3_1Display.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a3_1Display.ScaleTransferFunction.Points = [0.020840197801589966, 0.0, 0.5, 0.0, 0.8670650005340577, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a3_1Display.OpacityTransferFunction.Points = [0.020840197801589966, 0.0, 0.5, 0.0, 0.8670650005340577, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1LUT in view renderView7
pk1LUTColorBar_8 = GetScalarBar(pk1LUT, renderView7)
pk1LUTColorBar_8.Title = 'pk1'
pk1LUTColorBar_8.ComponentTitle = '0'

# set color bar visibility
pk1LUTColorBar_8.Visibility = 1

# show color legend
a3_1Display.SetScalarBarVisibility(renderView7, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView8'
# ----------------------------------------------------------------

# show data from a3_1
a3_1Display_1 = Show(a3_1, renderView8, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a3_1Display_1.Representation = 'Surface'
a3_1Display_1.ColorArrayName = ['POINTS', 'health']
a3_1Display_1.LookupTable = healthLUT
a3_1Display_1.SelectTCoordArray = 'None'
a3_1Display_1.SelectNormalArray = 'None'
a3_1Display_1.SelectTangentArray = 'None'
a3_1Display_1.OSPRayScaleArray = 'Eps'
a3_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a3_1Display_1.SelectOrientationVectors = 'None'
a3_1Display_1.ScaleFactor = 0.0029217138886451725
a3_1Display_1.SelectScaleArray = 'Eps_v'
a3_1Display_1.GlyphType = 'Arrow'
a3_1Display_1.GlyphTableIndexArray = 'Eps'
a3_1Display_1.GaussianRadius = 0.0001460856944322586
a3_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a3_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a3_1Display_1.OpacityArray = ['POINTS', 'Eps']
a3_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a3_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a3_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a3_1Display_1.ScalarOpacityFunction = healthPWF
a3_1Display_1.ScalarOpacityUnitDistance = 0.002590990536386772
a3_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a3_1Display_1.ScaleTransferFunction.Points = [0.020840197801589966, 0.0, 0.5, 0.0, 0.8670650005340577, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a3_1Display_1.OpacityTransferFunction.Points = [0.020840197801589966, 0.0, 0.5, 0.0, 0.8670650005340577, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for healthLUT in view renderView8
healthLUTColorBar_10 = GetScalarBar(healthLUT, renderView8)
healthLUTColorBar_10.Title = 'health'
healthLUTColorBar_10.ComponentTitle = ''

# set color bar visibility
healthLUTColorBar_10.Visibility = 1

# show color legend
a3_1Display_1.SetScalarBarVisibility(renderView8, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView9'
# ----------------------------------------------------------------

# show data from a4_1
a4_1Display_1 = Show(a4_1, renderView9, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a4_1Display_1.Representation = 'Surface'
a4_1Display_1.ColorArrayName = ['POINTS', 'pk1']
a4_1Display_1.LookupTable = pk1LUT
a4_1Display_1.SelectTCoordArray = 'None'
a4_1Display_1.SelectNormalArray = 'None'
a4_1Display_1.SelectTangentArray = 'None'
a4_1Display_1.OSPRayScaleArray = 'Eps'
a4_1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
a4_1Display_1.SelectOrientationVectors = 'None'
a4_1Display_1.ScaleFactor = 0.002933994494378567
a4_1Display_1.SelectScaleArray = 'Eps_v'
a4_1Display_1.GlyphType = 'Arrow'
a4_1Display_1.GlyphTableIndexArray = 'Eps'
a4_1Display_1.GaussianRadius = 0.00014669972471892834
a4_1Display_1.SetScaleArray = ['POINTS', 'Eps']
a4_1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
a4_1Display_1.OpacityArray = ['POINTS', 'Eps']
a4_1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
a4_1Display_1.DataAxesGrid = 'GridAxesRepresentation'
a4_1Display_1.PolarAxes = 'PolarAxesRepresentation'
a4_1Display_1.ScalarOpacityFunction = pk1PWF
a4_1Display_1.ScalarOpacityUnitDistance = 0.0025983743994128345
a4_1Display_1.OpacityArrayName = ['POINTS', 'Eps']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a4_1Display_1.ScaleTransferFunction.Points = [0.01342993974685669, 0.0, 0.5, 0.0, 0.7344027876853944, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a4_1Display_1.OpacityTransferFunction.Points = [0.01342993974685669, 0.0, 0.5, 0.0, 0.7344027876853944, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pk1LUT in view renderView9
pk1LUTColorBar_9 = GetScalarBar(pk1LUT, renderView9)
pk1LUTColorBar_9.Title = 'pk1'
pk1LUTColorBar_9.ComponentTitle = '0'

# set color bar visibility
pk1LUTColorBar_9.Visibility = 1

# show color legend
a4_1Display_1.SetScalarBarVisibility(renderView9, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(a0_1)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
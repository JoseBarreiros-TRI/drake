import numpy as np
import os
import sys
from urllib.request import urlretrieve
from IPython import get_ipython
import pdb 

from pydrake.all import (JointActuatorIndex, JointIndex, namedview)
import pydrake.all
from pydrake.all import (BallRpyJoint, Box, CoulombFriction, Cylinder,
                         PrismaticJoint, RevoluteJoint, RigidTransform,
                         SpatialInertia, Sphere, UnitInertia)
                         
from pydrake.multibody.tree import JointIndex
from pydrake.common.containers import namedview

# Use a global variable here because some calls to IPython will actually case an
# interpreter to be created.  This file needs to be imported BEFORE that
# happens.
running_as_notebook = "COLAB_TESTING" not in os.environ and get_ipython(
) and hasattr(get_ipython(), 'kernel')

running_as_test = False


def set_running_as_test(value):
    global running_as_test
    running_as_test = value


def pyplot_is_interactive():
    # import needs to happen after the backend is set.
    import matplotlib.pyplot as plt
    from matplotlib.rcsetup import interactive_bk
    return plt.get_backend() in interactive_bk


def FindResource(filename):
    return os.path.join(os.path.dirname(__file__), filename)


# A filename from the data directory.  Will download if necessary.
def LoadDataResource(filename):
    data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    if not os.path.exists(data):
        os.makedirs(data)
    path = os.path.join(data, filename)
    if not os.path.exists(path):
        urlretrieve(f"https://manipulation.csail.mit.edu/data/{filename}", path)
    return path


def AddPackagePaths(parser):
    # Remove once https://github.com/RobotLocomotion/drake/issues/10531 lands.
    parser.package_map().PopulateFromFolder(FindResource(""))
    parser.package_map().Add(
        "manipulation_station",
        os.path.join(pydrake.common.GetDrakePath(),
                     "examples/manipulation_station/models"))
    parser.package_map().Add(
        "ycb",
        os.path.join(pydrake.common.GetDrakePath(), "manipulation/models/ycb"))
    parser.package_map().Add(
        "wsg_50_description",
        os.path.join(pydrake.common.GetDrakePath(),
                     "manipulation/models/wsg_50_description"))


reserved_labels = [
    pydrake.geometry.render.RenderLabel.kDoNotRender,
    pydrake.geometry.render.RenderLabel.kDontCare,
    pydrake.geometry.render.RenderLabel.kEmpty,
    pydrake.geometry.render.RenderLabel.kUnspecified,
]


def colorize_labels(image):
    # import needs to happen after backend is set up.
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    """Colorizes labels."""
    # TODO(eric.cousineau): Revive and use Kuni's palette.
    cc = mpl.colors.ColorConverter()
    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = np.array([cc.to_rgb(c["color"]) for c in color_cycle])
    bg_color = [0, 0, 0]
    image = np.squeeze(image)
    background = np.zeros(image.shape[:2], dtype=bool)
    for label in reserved_labels:
        background |= image == int(label)
    foreground = image[np.logical_not(background)]
    color_image = colors[image % len(colors)]
    color_image[background] = bg_color
    return color_image


def SetupMatplotlibBackend(wishlist=["notebook"]):
    """
    Helper to support multiple workflows:
        1) nominal -- running locally w/ jupyter notebook
        2) unit tests (no ipython, backend is template)
        3) binder -- does have notebook backend
        4) colab -- claims to have notebook, but it doesn't work
    Puts the matplotlib backend into notebook mode, if possible,
    otherwise falls back to inline mode.
    Returns True iff the final backend is interactive.
    """
    # To find available backends, one can access the lists:
    # matplotlib.rcsetup.interactive_bk
    # matplotlib.rcsetup.non_interactive_bk
    # matplotlib.rcsetup.all_backends
    if running_as_notebook:
        ipython = get_ipython()
        # Short-circuit for google colab.
        if 'google.colab' in sys.modules:
            ipython.run_line_magic("matplotlib", "inline")
            return False
        # TODO: Find a way to detect vscode, and use inline instead of notebook
        for backend in wishlist:
            try:
                ipython.run_line_magic("matplotlib", backend)
                return pyplot_is_interactive()
            except KeyError:
                continue
        ipython.run_line_magic("matplotlib", "inline")
    return False


# TODO(russt): promote these to drake (and make a version with model_instance)


# Adapted from Drake's system_doxygen.py.  Just make an html rendering of the
# system block with its name and input/output ports (even if it is a Diagram).
def SystemHtml(system):
    input_port_html = ""
    for p in range(system.num_input_ports()):
        input_port_html += (
            f'<tr><td align=right style=\"padding:5px 0px 5px 0px\">'
            f'{system.get_input_port(p).get_name()} &rarr;</td></tr>')
    output_port_html = ""
    for p in range(system.num_output_ports()):
        output_port_html += (
            '<tr><td align=left style=\"padding:5px 0px 5px 0px\">'
            f'&rarr; {system.get_output_port(p).get_name()}</td></tr>')
    # Note: keeping this on a single line avoids having to handle comment line
    # markers (e.g. * or ///)
    html = (
        f'<table align=center cellpadding=0 cellspacing=0><tr align=center>'
        f'<td style=\"vertical-align:middle\">'
        f'<table cellspacing=0 cellpadding=0>{input_port_html}</table>'
        f'</td>'
        f'<td align=center style=\"border:2px solid black;padding-left:20px;'
        f'padding-right:20px;vertical-align:middle\" bgcolor=#F0F0F0>'
        f'{system.get_name()}</td>'
        f'<td style=\"vertical-align:middle\">'
        f'<table cellspacing=0 cellpadding=0>{output_port_html}</table>'
        f'</td></tr></table>')

    return html

def MakeNamedViewPositions(mbp, view_name, add_suffix_if_single_position=False):
    names = [None] * mbp.num_positions()
    for ind in range(mbp.num_joints()):
        joint = mbp.get_joint(JointIndex(ind))
        if joint.num_positions() == 1 and not add_suffix_if_single_position:
            names[joint.position_start()] = joint.name()
        else:
            for i in range(joint.num_positions()):
                names[joint.position_start() + i] = \
                    f"{joint.name()}_{joint.position_suffix(i)}"
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_positions_start()
        for i in range(7):
            names[start
                  + i] = f"{body.name()}_{body.floating_position_suffix(i)}"
    return namedview(view_name, names)


def MakeNamedViewVelocities(mbp,
                            view_name,
                            add_suffix_if_single_velocity=False):
    names = [None] * mbp.num_velocities()
    for ind in range(mbp.num_joints()):
        joint = mbp.get_joint(JointIndex(ind))
        if joint.num_velocities() == 1 and not add_suffix_if_single_velocity:
            names[joint.velocity_start()] = joint.name()
        else:
            for i in range(joint.num_velocities()):
                names[joint.velocity_start() + i] = \
                    f"{joint.name()}_{joint.velocity_suffix(i)}"
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_velocities_start() - mbp.num_positions()
        for i in range(6):
            names[start
                  + i] = f"{body.name()}_{body.floating_velocity_suffix(i)}"
    return namedview(view_name, names)


def MakeNamedViewState(mbp, view_name):
    # TODO: this could become a nested named view, pending
    # https://github.com/RobotLocomotion/drake/pull/14973
    pview = MakeNamedViewPositions(mbp, f"{view_name}_pos", True)
    vview = MakeNamedViewVelocities(mbp, f"{view_name}_vel", True)
    return namedview(view_name, pview.get_fields() + vview.get_fields())


def MakeNamedViewActuation(mbp, view_name):
    names = [None] * mbp.get_actuation_input_port().size()
   
    for ind in range(mbp.num_actuators()):
        joint = mbp.get_joint(JointIndex(ind))
        actuator=mbp.get_joint_actuator(JointActuatorIndex(ind))
        names[actuator.input_start()] = actuator.name()
    return namedview(view_name, names)

def AddShape(plant, shape, name, mass=1, mu=1, color=[.5, .5, .9, 1.0]):
    instance = plant.AddModelInstance(name)
    # TODO: Add a method to UnitInertia that accepts a geometry shape (unless
    # that dependency is somehow gross) and does this.
    if isinstance(shape, Box):
        inertia = UnitInertia.SolidBox(shape.width(), shape.depth(),
                                       shape.height())
    elif isinstance(shape, Cylinder):
        inertia = UnitInertia.SolidCylinder(shape.radius(), shape.length())
    elif isinstance(shape, Sphere):
        inertia = UnitInertia.SolidSphere(shape.radius())
    else:
        raise RunTimeError(
            f"need to write the unit inertia for shapes of type {shape}")
    body = plant.AddRigidBody(
        name, instance,
        SpatialInertia(mass=mass,
                       p_PScm_E=np.array([0., 0., 0.]),
                       G_SP_E=inertia))
    if plant.geometry_source_is_registered():
        if isinstance(shape, Box):
            plant.RegisterCollisionGeometry(
                body, RigidTransform(),
                Box(shape.width() - 0.001,
                    shape.depth() - 0.001,
                    shape.height() - 0.001), name, CoulombFriction(mu, mu))
            i = 0
            for x in [-shape.width() / 2.0, shape.width() / 2.0]:
                for y in [-shape.depth() / 2.0, shape.depth() / 2.0]:
                    for z in [-shape.height() / 2.0, shape.height() / 2.0]:
                        plant.RegisterCollisionGeometry(
                            body, RigidTransform([x, y, z]),
                            Sphere(radius=1e-7), f"contact_sphere{i}",
                            CoulombFriction(mu, mu))
                        i += 1
        else:
            plant.RegisterCollisionGeometry(body, RigidTransform(), shape, name,
                                            CoulombFriction(mu, mu))

        plant.RegisterVisualGeometry(body, RigidTransform(), shape, name, color)

    return instance
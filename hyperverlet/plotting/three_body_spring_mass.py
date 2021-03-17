from hyperverlet.energy import three_body_spring_mass
from hyperverlet.plotting.energy import plot_energy


def three_body_spring_mass_energy_plot(q, p, trajectory, m, k, l, plot_every=1):
    # Detatch and trim data
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    trajectory = trajectory.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()
    l = l.cpu().detach().numpy()
    k = k.cpu().detach().numpy()

    # Calculate energy of the system
    ke = three_body_spring_mass.calc_kinetic_energy(m, p)
    pe = three_body_spring_mass.calc_potential_energy(k, q, l)
    te = three_body_spring_mass.calc_total_energy(ke, pe)

    plot_energy(trajectory, te, ke, pe)

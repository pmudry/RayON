# About

### About the `RayON` project

`RayON` started as a clean-room reimplementation of the *Ray Tracing in One Weekend* series — a
personal exercise to understand the mechanics of stochastic light transport from first principles.
It grew into a full-featured interactive GPU path tracer used as an educational resource in the
*CS302 High-Performance Computing* course for the ISC bachelor programme at HES-SO Valais.

The project demonstrates:

- How one scene description drives two GPU back-ends, a software path tracer in CUDA and a
  hardware-accelerated one on OptiX, without duplicating the scene logic.
- Progressive sample accumulation and interactive camera control at real-time frame rates,
  with a regulator that trades resolution for frame rate while the camera moves.
- Physically-based materials, from Lambertian and microfacet conductors to thin films, clear
  coats and participating media, with next event estimation and multiple importance sampling.
- Ray-marched signed distance shapes and volumetric clouds from procedural noise or OpenVDB
  data.

The source code is published under the GPL v3 licence and is intended as a learning resource
as much as a working renderer.

## About the author

Pierre-André Mudry is head of [*Computer Science and Communication degree programme*](https://www.hevs.ch/isc) and a professor at the **University of Applied Sciences and Arts Western Switzerland
(HES-SO Valais-Wallis)** in Sion, where he teaches and conducts research in the Systems Engineering
department.

His work sits at the intersection of embedded systems, digital hardware design, and applied machine
learning. He has a particular interest in bridging the gap between algorithm and silicon — from
high-level language design for constrained devices through to FPGA implementation and IoT
deployment. Recent work has explored edge inference, non-invasive flow metering with neural
networks, and anomaly detection on resource-limited hardware.

### Research interests

- Embedded systems and processor architectures
- FPGA design and hardware/software co-design
- Internet of Things and edge computing
- Applied machine learning on constrained devices
- High-performance computing and GPU programming

---

- **Affiliation**: HES-SO Valais-Wallis — School of Engineering, Computer Science degree programme
- **GitHub**: [github.com/pmudry](https://github.com/pmudry)
- **LinkedIn**: [linkedin.com/in/pamudry](https://www.linkedin.com/in/pamudry)

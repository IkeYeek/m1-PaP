## Portfolio Projects — Master Year 1

I’m currently a first-year Master’s student in Computer Science at the University of Bordeaux. These repositories collect the projects I’ve worked on during my “Programmation des Architectures Parallèles”, Networking, Operating Systems and Software Engineering courses.  
Feel free to browse the code for inspiration—but please **do not copy** any part verbatim (plagiarism is taken very seriously, especially if you’re also a student here!).  

If you’d like to discuss any of these projects, need hints or feedback, or just want to geek out over HPC kernels or network protocols, drop me a line on [GitHub](https://github.com/ikeyeek) or send me an email at `luk@ike.icu`. I’m happy to help in my spare time!

---

## ⚙️ Game of Life (PAD — HPC Introduction)

**Description**  
Full source code for the “Programmation des Architectures Parallèles” (introduction to HPC) project: Conway’s Game of Life in 2D & 3D on a torus, with multiple branching kernels.

**My Contributions**  
- **OpenMP & OpenCL kernels**: Designed and optimized parallel loops.  
- **SIMD kernels**: Wrote AVX2 & AVX-512 versions (learnt the limitations of manual intrinsics vs. compiler auto-vectorization).  
- **Adaptive CPU/GPU scheduling**: Combined OpenMP & OpenCL to dynamically shift workload based on runtime performance.  
- **3D “life3D” on torus**: Outside syllabus—reproduced professor’s demo, including a cache-friendly Struct-of-Arrays GPU kernel.  

**What I Learned**  
- Hands-on experience with heterogeneous programming models (OpenMP, OpenCL).  
- Performance tuning: measuring, plotting and interpreting benchmarks with EasyPAP.  
- SIMD pitfalls: when manual intrinsics underperform compiler optimizations.  

<details>
<summary>Usage (coming soon…)</summary>

> *Instructions on how to build and run each branch will be added here soon. Feel free to contact me or open an issue if you need help in the meantime!*

</details>

> **License & Disclaimer**  
> Code is provided “as is” for educational use only. You may reference it, but **do not copy**—especially if you’re also at Uni Bordeaux (they definitly will get your ass and I won't do shit about it)! Instead, once again, ask me, still got a little bit of free time in my life!

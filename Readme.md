# About

A couple of years ago, I made a [software 3D renderer](https://github.com/mdepp/software-renderer) in C++.
While I was largely happy with the result, the code structure was poor and made adding interesting features like textures and shadows nearly impossible.
Moreover, its structure was quite different from existing software (e.g. OpenGL or DirectX) which effectively ruled out experimentation with techniques developed for those technologies.

This project is the beginnings of a re-write of that renderer to be easier to use.
I intend to loosely mimic some parts of OpenGL (e.g. its general transforms and coordinate spaces) but not others (e.g. the use of both right- and left-handed coordinate systems).
The goal is to enable experimentation with semi-modern rendering techniques in a way that is clearer and easier to write than using something like shaders in OpenGL.

What it it does so far:

![](docs/preview.gif)

#[macro_use]
extern crate glium;
extern crate glium_sdl2;
extern crate sdl2;
extern crate rand;

use glium::{Surface, texture};
use glium_sdl2::DisplayBuild;
use std::mem::swap;
use std::f64::consts::PI;
use rand::prelude::*;

const SCREEN_SIZE:     (u32, u32) = (1024, 768);
const SIMULATION_SIZE: (u32, u32) = (320, 240);

const NUMBER_OF_INKS:  u32 = 5;
const INK_DRAG:        f32 = 7.0;
const INK_SPREAD:      f32 = 50.0;
const INK_SIZE:        f32 = 15.0;
const VORTICITY:       f32 = 5.0;
const PRESSURE_ITERS:  u32 = 20;

const RED_RANGE:             (f32, f32) = (0.20, 0.90);
const GREEN_RANGE:           (f32, f32) = (0.15, 0.70);
const BLUE_RANGE:            (f32, f32) = (0.10, 0.50);
const COLOR_PERIOD_RANGE:    (f32, f32) = (10.0, 150.0);
const POSITION_PERIOD_RANGE: (f32, f32) = (20.0, 80.0);

const VERTEX_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;
	in vec2 position;
	out vec2 lpos, rpos, tpos, bpos, mpos;
	uniform vec2 texelsize;
	void main() {
		gl_Position = vec4(position.x, position.y, 0.0, 1.0);
		mpos = 0.5 * position + 0.5;
		lpos = mpos - vec2(texelsize.x, 0.0f);
		rpos = mpos + vec2(texelsize.x, 0.0f);
		bpos = mpos - vec2(0.0f, texelsize.y);
		tpos = mpos + vec2(0.0f, texelsize.y);
	}"#;

const INK_VERTEX_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;
	in vec2 position;
	out vec2 mpos;
	uniform vec2 ink_position;
	uniform float ink_size;

	out vec2 lpos, rpos, tpos, bpos;
	uniform vec2 texelsize;

	void main() {
		mpos = (ink_position + position * ink_size) * texelsize;

		lpos = mpos - vec2(texelsize.x, 0.0f);
		rpos = mpos + vec2(texelsize.x, 0.0f);
		bpos = mpos - vec2(0.0f, texelsize.y);
		tpos = mpos + vec2(0.0f, texelsize.y);

		gl_Position = vec4(2.0 * mpos - 1.0, 0.0, 1.0);
	}"#;

const ADVECTION_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;
	in vec2 mpos;
	uniform sampler2D image;
	uniform sampler2D velocity;
	out vec4 fragment;
	uniform vec2 texelsize;
	uniform float dt;

	void main() {
		vec2 vel = texture(velocity, mpos).xy;
		fragment = texture(image, mpos - vel * texelsize * dt);
	}"#;

const DIVERGENCE_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;
	in vec2 mpos;
	uniform sampler2D velocity;
	in vec2 lpos, rpos, tpos, bpos;
	out vec4 fragment;

	void main() {
		float l = texture(velocity, lpos).x;
		float r = texture(velocity, rpos).x;
		float t = texture(velocity, tpos).y;
		float b = texture(velocity, bpos).y;

		vec2 m = texture(velocity, mpos).xy;
		if(lpos.x < 0.0) {l = -m.x;}
		if(rpos.x > 1.0) {r = -m.x;}
		if(tpos.y > 1.0) {t = -m.y;}
		if(bpos.y < 0.0) {b = -m.y;}

		float div = 0.5 * (r - l + t - b);
		fragment = vec4(div, 0.0, 0.0, 1.0);
	}"#;

const PRESSURE_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;
	uniform sampler2D divergence;
	uniform sampler2D pressure;
	in vec2 lpos, rpos, tpos, bpos, mpos;
	out vec4 fragment;

	void main() {
		float l = texture(pressure, lpos).x;
		float r = texture(pressure, rpos).x;
		float t = texture(pressure, tpos).x;
		float b = texture(pressure, bpos).x;
		float m = texture(divergence, mpos).x;

		float p = 0.25 * (l + r + t + b - m);
		fragment = vec4(p, 0.0, 0.0, 1.0);
	}"#;


const DIFFERENCE_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;
	in vec2 mpos, lpos, rpos, tpos, bpos;
	uniform sampler2D pressure;
	uniform sampler2D velocity;
	out vec4 fragment;

	void main() {
		float l = texture(pressure, lpos).x;
		float r = texture(pressure, rpos).x;
		float t = texture(pressure, tpos).x;
		float b = texture(pressure, bpos).x;

		vec2 vel = texture(velocity, mpos).xy;
		vel.xy -= vec2(r - l, t - b) * 0.5 * 0.8;
		fragment = vec4(vel, 0.0, 1.0);
	}"#;

const CURL_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;
	in vec2 mpos, lpos, rpos, tpos, bpos;
	out vec4 fragment;
	uniform sampler2D velocity;

	void main () {
		float l = texture(velocity, lpos).y;
		float r = texture(velocity, rpos).y;
		float t = texture(velocity, tpos).x;
		float b = texture(velocity, bpos).x;
		float curl = 0.5 * (r - l - t + b);
		fragment = vec4(curl, 0.0, 0.0, 1.0);
	}"#;

const VORTICITY_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;
	in vec2 mpos, lpos, rpos, tpos, bpos;
	out vec4 fragment;
	uniform sampler2D velocity;
	uniform sampler2D curl;
	uniform float dt;
	uniform float vorticity;

	void main () {
		float l = texture(curl, lpos).x;
		float r = texture(curl, rpos).x;
		float t = texture(curl, tpos).x;
		float b = texture(curl, bpos).x;
		float m = texture(curl, mpos).x;

		vec2 force = 0.5 * vec2(abs(t) - abs(b), abs(l) - abs(r));
		force /= length(force) + 0.0001;
		force *= vorticity * m;

		vec2 velocity = texture(velocity, mpos).xy;
		velocity += force * dt;
		fragment = vec4(velocity, 0.0, 1.0);
	}"#;

const FORCE_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;

	in vec2 mpos;
	out vec4 fragment;

	uniform sampler2D velocity;

	uniform vec2 ink_position;
	uniform vec2 ink_velocity;
	uniform float ink_drag;
	uniform float ink_spread;
	uniform float ink_size;

	uniform vec2 texelsize;
	uniform float dt;

	mediump float rand(vec2 co)
	{
		mediump float a = 12.9898;
		mediump float b = 78.233;
		mediump float c = 43758.5453;
		mediump float dt= dot(co.xy ,vec2(a,b));
		mediump float sn= mod(dt,3.14);
		return fract(sin(sn) * c);
	}
	void main() {
		vec2 ink_ortho = vec2(ink_velocity.y, -ink_velocity.x);
		vec2 direction = (mpos / texelsize - ink_position);
		float distance = length(direction) + 1e-5;

		float fade = pow(max(1.0 - distance / ink_size, 0.0), 2.0);
		float dx = dot(ink_velocity, direction) / (length(ink_velocity) * ink_size + 1e-5);
		float dy = dot(ink_ortho, direction) / (length(ink_ortho) * ink_size + 1e-5);

		vec2 drag = ((1.0 - dx*dx - dy*dy) * ink_velocity - 2.0 * dx*dy * ink_ortho) * ink_drag;
		vec2 spread = direction / distance * ink_spread * 100.0;
		vec2 vel = drag + spread;
		fragment = vec4(vel, 0.0, fade * dt);
	}"#;

const INK_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;

	in vec2 mpos;
	out vec4 fragment;

	uniform vec2 ink_position;
	uniform vec4 ink_color;
	uniform float ink_size;
	uniform vec2 texelsize;

	void main() {
		vec2 direction = (ink_position - mpos / texelsize);
		float distance = length(direction);
		float fade = pow(max(1.0 - length(direction) / ink_size * 3.0, 0.0), 0.3);
		vec4 icolor = pow(ink_color, vec4(2.2));
		fragment = vec4(icolor.r, icolor.g, icolor.b, fade);
	}"#;

const OUTPUT_SHADER_SOURCE: &str = r#"#version 300 es
	precision mediump float;
	in vec2 mpos;
	out vec4 fragment;
	uniform sampler2D image;
	void main() {
		fragment = texture(image, mpos);
		fragment = pow(fragment, vec4(1.0 / 2.2));
	}"#;

#[derive(Copy, Clone)]
struct Vertex {
	position: [f32; 2]
}

implement_vertex!(Vertex, position);

fn empty_data(size : (u32, u32)) -> texture::RawImage2d<'static, f32>
{
	let pixels : Vec<f32> = vec![0.0; (size.0 * size.1 * 4) as usize];
	return texture::RawImage2d::from_raw_rgba(pixels, size);
}

#[derive(Debug)]
struct InkState
{
	position : (f32, f32),
	velocity : (f32, f32),
	color	: (f32, f32, f32, f32)
}

struct Simulation {
	sdl_context : sdl2::Sdl,
	display     : glium_sdl2::Display,

	timer : f64,
	last_frame		   : std::time::Instant,
	vertex_buffer	   : glium::VertexBuffer<Vertex>,
	index_buffer	   : glium::IndexBuffer<u16>,

	advection_program  : glium::Program,
	divergence_program : glium::Program,
	pressure_program   : glium::Program,
	difference_program : glium::Program,
	curl_program	   : glium::Program,
	vorticity_program  : glium::Program,
	force_program	   : glium::Program,
	ink_program		   : glium::Program,
	output_program	   : glium::Program,

	color_texture		  : texture::Texture2d,
	color_texture_work	  : texture::Texture2d,
	velocity_texture	  : texture::Texture2d,
	velocity_texture_work : texture::Texture2d,
	divergence_texture	  : texture::Texture2d,
	curl_texture		  : texture::Texture2d,
	pressure_texture	  : texture::Texture2d,
	pressure_texture_work : texture::Texture2d,

	inks				  : Vec::<InkParameter>
}

struct Oscillation {
	period : f64,
	phase : f64,
	range : (f64, f64)
}

impl Oscillation {
	fn value_at_time(&self, time : f64) -> f32 {
		((time / self.period * 2.0 * PI + self.phase).cos()
			* (self.range.1 - self.range.0) * 0.5 + (self.range.1 + self.range.0) * 0.5) as f32
	}

	fn derivate_at_time(&self, time : f64) -> f32 {
		((time / self.period * 2.0 * PI + self.phase).sin()
			* (self.range.1 - self.range.0) * 0.5
			* (-1.0 / self.period * 2.0 * PI)) as f32
	}
}

struct PositionParameter {
	xosc : Oscillation,
	yosc : Oscillation
}

impl PositionParameter {
	fn randomize() -> PositionParameter {
		let mut rng = thread_rng();
		let inbalance = 2.0;
		let xmin = ( rng.gen::<f64>().powf(inbalance)) * (SIMULATION_SIZE.0 as f64);
		let xmax = (-rng.gen::<f64>().powf(inbalance) + 1.0) * (SIMULATION_SIZE.0 as f64);
		let ymin = ( rng.gen::<f64>().powf(inbalance)) * (SIMULATION_SIZE.1 as f64);
		let ymax = (-rng.gen::<f64>().powf(inbalance) + 1.0) * (SIMULATION_SIZE.1 as f64);

		PositionParameter {
			xosc: Oscillation {
				period: rng.gen_range(POSITION_PERIOD_RANGE.0, POSITION_PERIOD_RANGE.1) as f64,
				phase: rng.gen_range(0.0, 2.0 * PI),
				range: (xmin, xmax)
			},
			yosc: Oscillation {
				period: rng.gen_range(POSITION_PERIOD_RANGE.0, POSITION_PERIOD_RANGE.1) as f64,
				phase: rng.gen_range(0.0, 2.0 * PI),
				range: (ymin, ymax)
			}
		}
	}
}

struct ColorParameter {
	rosc : Oscillation,
	gosc : Oscillation,
	bosc : Oscillation
}

impl ColorParameter {
	fn randomize() -> ColorParameter {
		let mut rng = thread_rng();
		ColorParameter {
			rosc: Oscillation {
				period: rng.gen_range(COLOR_PERIOD_RANGE.0 as f64, COLOR_PERIOD_RANGE.1 as f64),
				phase: rng.gen_range(0.0, 2.0 * PI),
				range: (RED_RANGE.0 as f64, RED_RANGE.1 as f64)
			},
			gosc: Oscillation {
				period: rng.gen_range(COLOR_PERIOD_RANGE.0 as f64, COLOR_PERIOD_RANGE.1 as f64),
				phase: rng.gen_range(0.0, 2.0 * PI),
				range: (GREEN_RANGE.0 as f64, GREEN_RANGE.1 as f64)
			},
			bosc: Oscillation {
				period: rng.gen_range(COLOR_PERIOD_RANGE.0 as f64, COLOR_PERIOD_RANGE.1 as f64),
				phase: rng.gen_range(0.0, 2.0 * PI),
				range: (BLUE_RANGE.0 as f64, BLUE_RANGE.1 as f64)
			}
		}
	}
}

struct InkParameter {
	position : PositionParameter,
	color : ColorParameter
}

impl InkParameter {
	fn randomize() -> InkParameter {
		InkParameter {
			position: PositionParameter::randomize(),
			color: ColorParameter::randomize()
		}
	}

	fn state_at_time(&self, time : f64) -> InkState {
		let xpos = self.position.xosc.value_at_time(time);
		let ypos = self.position.yosc.value_at_time(time);
		let xvel = self.position.xosc.derivate_at_time(time);
		let yvel = self.position.yosc.derivate_at_time(time);
		let rcolor = self.color.rosc.value_at_time(time);
		let gcolor = self.color.gosc.value_at_time(time);
		let bcolor = self.color.bosc.value_at_time(time);

		InkState {
			position: (xpos, ypos),
			velocity: (xvel, yvel),
			color: (rcolor, gcolor, bcolor, 1.0)}
	}
}

impl Simulation
{
	pub fn new() -> Simulation
	{
		let sdl_context = sdl2::init().unwrap();
		let video_subsystem = sdl_context.video().unwrap();
		let gl_attr = video_subsystem.gl_attr();
		gl_attr.set_context_version(3, 0);
		gl_attr.set_context_profile(sdl2::video::GLProfile::GLES);
		let display = video_subsystem.window("Fluid simulation", SCREEN_SIZE.0, SCREEN_SIZE.1)
			//~ .fullscreen()
			.build_glium()
			.unwrap();

		let vertices = vec![
			Vertex{position: [-1.0, -1.0]},
			Vertex{position: [ 1.0, -1.0]},
			Vertex{position: [ 1.0,  1.0]},
			Vertex{position: [-1.0,  1.0]}];

		let mut inks = Vec::<InkParameter>::new();
		for _i in 0..NUMBER_OF_INKS {
			inks.push(InkParameter::randomize());
		}

		return Simulation {
			sdl_context,
			last_frame: std::time::Instant::now(),
			timer: 0.0,

			vertex_buffer: glium::VertexBuffer::new(&display, &vertices).unwrap(),
			index_buffer: glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TriangleFan, &[0u16, 1, 2, 3]).unwrap(),

			advection_program:  glium::Program::from_source(&display, VERTEX_SHADER_SOURCE, ADVECTION_SHADER_SOURCE, None).unwrap(),
			divergence_program: glium::Program::from_source(&display, VERTEX_SHADER_SOURCE, DIVERGENCE_SHADER_SOURCE, None).unwrap(),
			pressure_program:   glium::Program::from_source(&display, VERTEX_SHADER_SOURCE, PRESSURE_SHADER_SOURCE, None).unwrap(),
			difference_program: glium::Program::from_source(&display, VERTEX_SHADER_SOURCE, DIFFERENCE_SHADER_SOURCE, None).unwrap(),
			curl_program:	   glium::Program::from_source(&display, VERTEX_SHADER_SOURCE, CURL_SHADER_SOURCE, None).unwrap(),
			vorticity_program:  glium::Program::from_source(&display, VERTEX_SHADER_SOURCE, VORTICITY_SHADER_SOURCE, None).unwrap(),
			force_program:	  glium::Program::from_source(&display, INK_VERTEX_SHADER_SOURCE, FORCE_SHADER_SOURCE, None).unwrap(),
			ink_program:		glium::Program::from_source(&display, INK_VERTEX_SHADER_SOURCE, INK_SHADER_SOURCE, None).unwrap(),
			output_program:	 glium::Program::from_source(&display, VERTEX_SHADER_SOURCE, OUTPUT_SHADER_SOURCE, None).unwrap(),

			color_texture:		 texture::Texture2d::with_format(&display, empty_data(SCREEN_SIZE),
				texture::UncompressedFloatFormat::F32F32F32F32, texture::MipmapsOption::NoMipmap).unwrap(),
			color_texture_work:	texture::Texture2d::with_format(&display, empty_data(SCREEN_SIZE),
				texture::UncompressedFloatFormat::F32F32F32F32, texture::MipmapsOption::NoMipmap).unwrap(),
			velocity_texture:	  texture::Texture2d::with_format(&display, empty_data(SIMULATION_SIZE),
				texture::UncompressedFloatFormat::F32F32F32F32, texture::MipmapsOption::NoMipmap).unwrap(),
			velocity_texture_work: texture::Texture2d::with_format(&display, empty_data(SIMULATION_SIZE),
				texture::UncompressedFloatFormat::F32F32F32F32, texture::MipmapsOption::NoMipmap).unwrap(),
			divergence_texture:	texture::Texture2d::with_format(&display, empty_data(SIMULATION_SIZE),
				texture::UncompressedFloatFormat::F32F32F32F32, texture::MipmapsOption::NoMipmap).unwrap(),
			curl_texture:		  texture::Texture2d::with_format(&display, empty_data(SIMULATION_SIZE),
				texture::UncompressedFloatFormat::F32F32F32F32, texture::MipmapsOption::NoMipmap).unwrap(),
			pressure_texture:	  texture::Texture2d::with_format(&display, empty_data(SIMULATION_SIZE),
				texture::UncompressedFloatFormat::F32F32F32F32, texture::MipmapsOption::NoMipmap).unwrap(),
			pressure_texture_work: texture::Texture2d::with_format(&display, empty_data(SIMULATION_SIZE),
				texture::UncompressedFloatFormat::F32F32F32F32, texture::MipmapsOption::NoMipmap).unwrap(),
			display,
			inks
		}
	}


	pub fn run(&mut self)
	{
		let dt = std::cmp::min(self.last_frame.elapsed(), std::time::Duration::from_millis(30)).as_secs_f32();
		//~ println!("FPS = {}", 1.0 / self.last_frame.elapsed().as_secs_f32());

		self.last_frame = std::time::Instant::now();
		self.timer += dt as f64;

		let simulation_viewport = glium::Rect{left: 0, bottom: 0, width: SIMULATION_SIZE.0, height: SIMULATION_SIZE.1};
		let simulation_parameters = glium::DrawParameters {viewport: Some(simulation_viewport), .. Default::default()};
		let blend_parameters = glium::DrawParameters {blend: glium::Blend::alpha_blending(), .. Default::default()};
		let texelsize = (1.0 / (SIMULATION_SIZE.0 as f32), 1.0 / (SIMULATION_SIZE.1 as f32));

		let ink_states : Vec<InkState> = self.inks.iter().map(|x| x.state_at_time(self.timer)).collect();
		fn boudary_sampler(texture : &texture::Texture2d) -> glium::uniforms::Sampler<texture::Texture2d> {
			glium::uniforms::Sampler::new(texture)
			.magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
			.minify_filter(glium::uniforms::MinifySamplerFilter::Linear)
			.wrap_function(glium::uniforms::SamplerWrapFunction::Clamp)};

		self.curl_texture.as_surface().draw(&self.vertex_buffer, &self.index_buffer, &self.curl_program,
			&uniform!{
				velocity: boudary_sampler(&self.velocity_texture),
				texelsize: texelsize},
			&simulation_parameters).unwrap();

		self.velocity_texture_work.as_surface().draw(&self.vertex_buffer, &self.index_buffer, &self.vorticity_program,
			&uniform!{
				velocity: boudary_sampler(&self.velocity_texture),
				curl: boudary_sampler(&self.curl_texture),
				texelsize: texelsize, vorticity: VORTICITY, dt: dt},
			&simulation_parameters).unwrap();
		swap(&mut self.velocity_texture, &mut self.velocity_texture_work);

		self.divergence_texture.as_surface().draw(&self.vertex_buffer, &self.index_buffer, &self.divergence_program,
			&uniform!{
				velocity: boudary_sampler(&self.velocity_texture),
				texelsize: texelsize},
			&simulation_parameters).unwrap();

		for _i in 1..PRESSURE_ITERS {
			self.pressure_texture_work.as_surface().draw(&self.vertex_buffer, &self.index_buffer, &self.pressure_program,
				&uniform!{
					divergence: boudary_sampler(&self.divergence_texture),
					pressure: boudary_sampler(&self.pressure_texture),
					texelsize: texelsize},
				&simulation_parameters).unwrap();
			swap(&mut self.pressure_texture, &mut self.pressure_texture_work);
		}

		self.velocity_texture_work.as_surface().draw(&self.vertex_buffer, &self.index_buffer, &self.difference_program,
			&uniform!{velocity: boudary_sampler(&self.velocity_texture), pressure: boudary_sampler(&self.pressure_texture), texelsize: texelsize},
			&simulation_parameters).unwrap();
		swap(&mut self.velocity_texture, &mut self.velocity_texture_work);

		for ink in ink_states.iter() {
			self.velocity_texture.as_surface().draw(&self.vertex_buffer, &self.index_buffer, &self.force_program,
				&uniform!{
					ink_position: ink.position,
					ink_velocity: ink.velocity,
					ink_size: INK_SIZE,
					ink_drag: INK_DRAG,
					ink_spread: INK_SPREAD,
					velocity: boudary_sampler(&self.velocity_texture),
					texelsize: texelsize,
					dt: dt},
				&glium::DrawParameters {
					viewport: Some(simulation_viewport),
					blend: glium::Blend::alpha_blending(),
					.. Default::default()}).unwrap();}

		for ink in ink_states.iter() {
			self.color_texture.as_surface().draw(&self.vertex_buffer, &self.index_buffer, &self.ink_program,
				&uniform!{
					ink_position: ink.position,
					ink_color: ink.color,
					ink_size: INK_SIZE,
					texelsize: texelsize},
				&blend_parameters).unwrap();}

		self.velocity_texture_work.as_surface().draw(&self.vertex_buffer, &self.index_buffer, &self.advection_program,
			&uniform!{
				image: boudary_sampler(&self.velocity_texture),
				velocity: boudary_sampler(&self.velocity_texture),
				texelsize: texelsize, dt: dt},
			&simulation_parameters).unwrap();
		swap(&mut self.velocity_texture, &mut self.velocity_texture_work);

		self.color_texture_work.as_surface().draw(&self.vertex_buffer, &self.index_buffer, &self.advection_program,
			&uniform!{
				image: boudary_sampler(&self.color_texture),
				velocity: boudary_sampler(&self.velocity_texture),
				texelsize: texelsize, dt: dt},
			&Default::default()).unwrap();
		swap(&mut self.color_texture, &mut self.color_texture_work);


		let mut target = self.display.draw();
		target.draw(&self.vertex_buffer, &self.index_buffer, &self.output_program,
			&uniform!{image: boudary_sampler(&self.color_texture), texelsize: texelsize}, &Default::default()).unwrap();
		target.finish().unwrap();
	}
}

impl emscripten_main_loop::MainLoop for Simulation
{
	fn main_loop(&mut self) -> emscripten_main_loop::MainLoopEvent
	{
		let mut running = emscripten_main_loop::MainLoopEvent::Continue;
		let mut event_pump = self.sdl_context.event_pump().unwrap();

		self.run();
		for event in event_pump.poll_iter() {
			use sdl2::event::Event;

			match event {
				Event::Quit { .. } => {
					running = emscripten_main_loop::MainLoopEvent::Terminate;
				},
				Event::KeyDown { keycode: Some(sdl2::keyboard::Keycode::Escape), .. } => {
					running = emscripten_main_loop::MainLoopEvent::Terminate;
				},
				_ => ()
			}
		}
		running
	}
}

fn main() {
	let simulation = Simulation::new();
	emscripten_main_loop::run(simulation);
}
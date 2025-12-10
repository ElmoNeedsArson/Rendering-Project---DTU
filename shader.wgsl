struct Uniforms {
    aspect: f32,
    cam_const: f32,
    jitterindex: u32,
    width: u32,
    eye: vec3f,
    height: u32,
    b1: vec3f,
    frame: u32,
    b2: vec3f,
    blue_background: u32,
    v: vec3f,
    gamma: f32,
    aperture_radius: f32,
    focal_distance: f32,
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32,
}

struct HitInfo {
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,
    color: vec3f,
    emission: vec3f,
    diffuse: vec3f,
    specular: vec3f,
    shader: u32,
    n1n2: f32,
    sphere_shininess: f32,
    tex_coords: vec2f,
    face_idx: u32,
    emit: bool,
    throughput: vec3f,
    sigma_t: vec3f
}

struct Onb {
    tangent: vec3f,
    binormal: vec3f,
    normal: vec3f,
}

struct Light {
    L_i: vec3f,
    w_i: vec3f,
    dist: f32
}

struct Material {
    emission: vec3f,
    diffuse: vec3f,
}

struct Aabb {
    min: vec3f,
    max: vec3f,
}

struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords: vec2f,
}

struct FSOut {
    @location(0) frame: vec4f,
    @location(1) accum: vec4f,
}

struct v_attribs {
    vPositions: vec3f,
    vertexNormals: vec3f
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;
@group(0) @binding(1)
var renderTexture: texture_2d<f32>;
@group(0) @binding(3)
var<storage> vAttribs: array<v_attribs>;
@group(0) @binding(4)
var<storage> meshFaces: array<vec4u>;
@group(0) @binding(6)
var<storage> materials: array<Material>;
@group(0) @binding(7)
var<storage> lightIndices: array<u32>;
@group(0) @binding(8)
var<uniform> aabb: Aabb;
@group(0) @binding(9)
var<storage> treeIds: array<u32>;
@group(0) @binding(10)
var<storage> bspTree: array<vec4u>;
@group(0) @binding(11)
var<storage> bspPlanes: array<f32>;

const MAX_LEVEL = 20u;
const BSP_LEAF = 3u;

var<private> branch_node: array<vec2u, MAX_LEVEL>;
var<private> branch_ray: array<vec2f, MAX_LEVEL>;

fn sample_point_light(pos: vec3f) -> Light {
    let light_position = vec3f(0.0, 1.0, 0.0);
    let rgb_intensity = vec3f(3.14, 3.14, 3.14);

    let l_i = rgb_intensity / (length(light_position - pos) * length(light_position - pos));
    let w_i = normalize(light_position - pos);
    let dist = distance(light_position, pos);

    return Light(l_i, w_i, dist);
}

fn sample_directional_light(pos: vec3f) -> Light {
    const l_e = vec3f(3.14, 3.14, 3.14);
    let w_e = normalize(vec3f(- 1.0));
    const dist = 1.0e32;
    return Light(l_e, - w_e, dist);
}

// fn sample_area_light(pos: vec3f, t: ptr<function, u32>) -> Light {
//     var intensity = vec3f(0);
//     var sum = vec3f(0);

//     for (var i = 0u; i < arrayLength(&lightIndices); i++) {
//         //let light_position = vPositions[lightIndeces[i]];

//         let v0 = vAttribs[meshFaces[lightIndices[i]].x].vPositions;
//         let v1 = vAttribs[meshFaces[lightIndices[i]].y].vPositions;
//         let v2 = vAttribs[meshFaces[lightIndices[i]].z].vPositions;

//         sum += (v0 + v1 + v2) / 3;
//     }

//     let light_position = sum / f32(arrayLength(&lightIndices));
//     let w_i = normalize(light_position - pos);

//     for (var i = 0u; i < arrayLength(&lightIndices); i++) {
//         let v0 = vAttribs[meshFaces[lightIndices[i]].x].vPositions;
//         let v1 = vAttribs[meshFaces[lightIndices[i]].y].vPositions;
//         let v2 = vAttribs[meshFaces[lightIndices[i]].z].vPositions;
//         let e0 = v1 - v0;
//         let e1 = v2 - v0;

//         var n = cross(e0, e1);
//         let area = length(n) * 0.5;
//         n = normalize(n);

//         intensity += abs(dot(- w_i, n)) * area * materials[meshFaces[lightIndices[i]].w].emission;
//     }

//     let l_i = intensity / (length(light_position - pos) * length(light_position - pos));
//     let dist = distance(light_position, pos);
//     //let dist  = length(light_position - pos);
//     return Light(l_i, w_i, dist);
// }

fn sample_area_light(pos: vec3f, t: ptr<function, u32>) -> Light {
    // Randomly pick a triangle from the light mesh
    let num_lights = arrayLength(&lightIndices);
    let rand_tri = u32(floor(rnd(t) * f32(num_lights)));
    let face_idx = lightIndices[rand_tri];

    // Get triangle vertices
    let v0 = vAttribs[meshFaces[face_idx].x].vPositions;
    let v1 = vAttribs[meshFaces[face_idx].y].vPositions;
    let v2 = vAttribs[meshFaces[face_idx].z].vPositions;

    // Sample barycentric coordinates for uniform sampling
    let xi1 = rnd(t);
    let xi2 = rnd(t);
    let sqrt_xi1 = sqrt(xi1);
    let alpha = 1.0 - sqrt_xi1;
    let beta = (1.0 - xi2) * sqrt_xi1;
    let gamma = xi2 * sqrt_xi1;

    // Compute random point on the triangle
    let light_position = alpha * v0 + beta * v1 + gamma * v2;

    // Compute interpolated normal
    let n0 = vAttribs[meshFaces[face_idx].x].vertexNormals;
    let n1 = vAttribs[meshFaces[face_idx].y].vertexNormals;
    let n2 = vAttribs[meshFaces[face_idx].z].vertexNormals;
    var n = normalize(alpha * n0 + beta * n1 + gamma * n2);

    // Compute area of the triangle
    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let area = 0.5 * length(cross(e0, e1));

    // Direction toward the sampled light point
    let w_i = normalize(light_position - pos);
    let dist = distance(light_position, pos);

    // Emitted radiance from the material (light color)
    let L_e = materials[meshFaces[face_idx].w].emission;

    // Compute contribution, weighted by cosine term and pdf
    // pdf(x) = 1 / (n * A_triangle)
    let pdf = 1.0 / (f32(num_lights) * area);
    let cos_theta_l = max(dot(- w_i, n), 0.0);
    let L_i = L_e * cos_theta_l / (dist * dist * pdf);

    return Light(L_i, w_i, dist);
}

fn intersect_plane(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, onb: Onb) -> bool {
    let denom = dot(r.direction, onb.normal);
    if (abs(denom) > 0.0001) {
        let dist = dot(position - r.origin, onb.normal) / denom;

        if (dist >= r.tmin && dist <= r.tmax) {
            hit.has_hit = true;
            hit.dist = dist;
            hit.normal = onb.normal;
            hit.position = r.origin + (dist * r.direction);
            // hit.specular = vec3f(0, 0, 0);
            // hit.ambient = hit.color*0.1;
            // hit.diffuse = hit.color*0.9;
            return true;
        }
    }

    return false;
}

//fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, v: array<vec3f, 3>) -> bool {
fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, face: u32) -> bool {
    // let e0 = v[1] - v[0];
    // let e1 = v[2] - v[0];
    let e0 = vAttribs[meshFaces[face].y].vPositions - vAttribs[meshFaces[face].x].vPositions;
    let e1 = vAttribs[meshFaces[face].z].vPositions - vAttribs[meshFaces[face].x].vPositions;
    let normal = cross(e0, e1);
    let denom = dot(normal, r.direction);
    if abs(denom) > 1e-8 {
        let tprime = (dot(normal, vAttribs[meshFaces[face].x].vPositions - r.origin)) / denom;
        if (tprime < r.tmax) && (tprime > r.tmin) {
            let c = cross((vAttribs[meshFaces[face].x].vPositions - r.origin), r.direction);
            let beta = dot(e1, c) / denom;
            if (beta < 0.0) {
                return false;
            }
            let gamma = - 1 * dot(e0, c) / denom;
            if (gamma < 0.0) || (beta + gamma > 1.0) {
                return false;
            }
            let alpha = 1.0 - beta - gamma;
            hit.has_hit = true;
            hit.dist = tprime;

            //hit.normal = normalize(normal);

            // Interpolate vertex normals using barycentric coordinates
            let n0 = vAttribs[meshFaces[face].x].vertexNormals;
            let n1 = vAttribs[meshFaces[face].y].vertexNormals;
            let n2 = vAttribs[meshFaces[face].z].vertexNormals;

            hit.normal = normalize(alpha * n0 + beta * n1 + gamma * n2);

            hit.position = r.origin + (tprime * r.direction);
            hit.specular = vec3f(0, 0, 0);
            hit.face_idx = face;
            return true;
        }
    }
    return false;
}

fn intersect_sphere(r: Ray, hit: ptr<function, HitInfo>, center: vec3f, radius: f32) -> bool {
    //⋯
    let a = dot(r.direction, r.direction);
    let b = 2.0 * dot(r.direction, r.origin - center);
    let c = dot(r.origin - center, r.origin - center) - radius * radius;
    if ((b / 2) * (b / 2) - c < 0) {
        return false;
    }
    let dist1 = - b / 2 - sqrt((b / 2) * (b / 2) - c);
    let dist2 = - b / 2 + sqrt((b / 2) * (b / 2) - c);
    if (dist1 >= r.tmin && dist1 <= r.tmax) {
        hit.has_hit = true;
        hit.dist = dist1;
        hit.normal = normalize((r.origin + dist1 * r.direction) - center);
        hit.position = r.origin + (dist1 * r.direction);
        hit.n1n2 = 1.0 / 1.5;
        hit.specular = vec3f(0.1, 0.1, 0.1);
        return true;
    }
    else if (dist2 >= r.tmin && dist2 <= r.tmax) {
        hit.has_hit = true;
        hit.dist = dist2;
        hit.normal = normalize((r.origin + dist2 * r.direction) - center);
        hit.position = r.origin + (dist2 * r.direction);
        hit.n1n2 = 1.0 / 1.5;
        hit.specular = vec3f(0.1, 0.1, 0.1);
        return true;
    }

    return false;
}

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, t: ptr<function, u32>) -> vec3f {
    //let light = sample_point_light(hit.position);
    var light = sample_area_light(hit.position, t);
    //let light = sample_directional_light(hit.position);

    // Ambient and emission
    let Le: vec3f = hit.emission;
    //let La: vec3f = hit.emission;

    // Create a shadow ray from hit.position towards the light
    let shadow_origin = hit.position + 0.001 * light.w_i;
    // offset to avoid self-intersection
    var shadow_ray = Ray(shadow_origin, light.w_i, 0.001, light.dist - 0.01);
    var shadow_hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 0u, 0f, 0f, vec2f(0.0, 0.0), 0u, true, vec3f(1.0), vec3f(0.0));

    // Check if the shadow ray hits anything before the light
    let in_shadow = intersect_scene(&shadow_ray, & shadow_hit);

    // if (in_shadow) {
    //     // Only ambient if in shadow
    //     return Le;
    // }
    // Diffuse shading - this is just the direct light
    var Lo: vec3f = (hit.diffuse / 3.14) * light.L_i * max(dot(hit.normal, light.w_i), 0.0);

    if (in_shadow) {
        // Only ambient if in shadow
        Lo = vec3f(0.0, 0.0, 0.0);
    }

    if (hit.emit) {
        Lo += Le;
    }

    //here comes new code:
    let cos_theta = sqrt(1 - rnd(t));
    let phi = 2.0 * 3.14 * rnd(t);
    let sin_theta = sqrt(1 - cos_theta * cos_theta);
    let temp = spherical_direction(sin_theta, cos_theta, phi);
    r.direction = rotate_to_normal(hit.normal, temp);
    r.origin = hit.position;
    r.tmin = 0.01;
    r.tmax = 10000.0;
    hit.has_hit = false;
    hit.throughput *= hit.diffuse;
    hit.emit = false;

    let roulette = rnd(t);
    let Pd = (hit.throughput.x + hit.throughput.y + hit.throughput.z) / 3;
    if (roulette < Pd) {
        hit.throughput /= Pd;
    }
    else {
        hit.has_hit = true;
    }

    //return Lo;
    return Lo;
}

// Given spherical coordinates, where theta is the polar angle and phi is the
// azimuthal angle, this function returns the corresponding direction vector
fn spherical_direction(sin_theta: f32, cos_theta: f32, phi: f32) -> vec3f {
    return vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Given a direction vector v sampled around the z-axis of a local coordinate system,
// this function applies the same rotation to v as is needed to rotate the z-axis to
// the actual direction n that v should have been sampled around
// [Frisvad, Journal of Graphics Tools 16, 2012;
// Duff et al., Journal of Computer Graphics Techniques 6, 2017].
fn rotate_to_normal(n: vec3f, v: vec3f) -> vec3f {
    let s = sign(n.z + 1.0e-16f);
    let a = - 1.0f / (1.0f + abs(n.z));
    let b = n.x * n.y * a;
    return vec3f(1.0f + n.x * n.x * a, b, - s * n.x) * v.x + vec3f(s * b, s * (1.0f + n.y * n.y * a), - n.y) * v.y + n * v.z;
}

fn phong(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    //need to calculate if ray is from outside or inside the sphere
    let light = sample_point_light(hit.position);
    let rho_d = hit.diffuse;
    let rho_s = vec3f(0.1, 0.1, 0.1);
    let s = hit.sphere_shininess;
    let w_o = - r.direction;
    let w_i = light.w_i;
    let n = hit.normal;
    let w_r = reflect(- w_i, hit.normal);

    let Lr = ((rho_d / 3.14) + (rho_s * ((s + 2) / 6.28) * pow(max(dot(w_r, w_o), 0.0), s))) * light.L_i * max(dot(n, w_i), 0.0);
    return Lr;
}

fn glossy(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let ph = phong(r, hit);
    refractive(r, hit);
    return ph;
}

fn fresnel_R(cos_theta_i: f32, cos_theta_t: f32, n_i: f32, n_t: f32) -> f32 {
    var r_perp = (n_i * cos_theta_i - n_t * cos_theta_t) / (n_i * cos_theta_i + n_t * cos_theta_t);
    var r_par = (n_t * cos_theta_i - n_i * cos_theta_t) / (n_t * cos_theta_i + n_i * cos_theta_t);
    var R = 0.5 * (r_perp * r_perp + r_par * r_par);
    return R;
}

// fn fresnel_R(cos_theta_i: f32, cos_theta_t: f32, ni_nt: f32) -> f32 {
//     let r_parl = (ni_nt * cos_theta_i - cos_theta_t) / (ni_nt * cos_theta_i + cos_theta_t);
//     let r_perp = (cos_theta_i - ni_nt * cos_theta_t) / (cos_theta_i + ni_nt * cos_theta_t);
//     return 0.5f * (r_parl * r_parl + r_perp * r_perp);
// }

fn fresnel(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, t: ptr<function, u32>) -> vec3f {
    //logic for a fresnel shader
    var cos_i = dot(r.direction, hit.normal);
    var R: f32;
    var n1 = 1.0;
    var n2 = 1.5;
    if (cos_i < 0.0) {
        //ray from outside the sphere
        hit.n1n2 = n1 / n2;
        cos_i = - cos_i;

        let cos2_t = 1 - ((hit.n1n2 * hit.n1n2) * (1 - (cos_i * cos_i)));

        R = fresnel_R(cos_i, sqrt(cos2_t), n1, n2);
        if (cos2_t < 0.0) {
            R = 1;
        }
    }
    else {
        //ray from inside the sphere
        hit.n1n2 = n2 / n1;
        hit.normal = - hit.normal;

        hit.throughput = exp(- hit.sigma_t * hit.dist);
        let prob = (hit.throughput.x + hit.throughput.y + hit.throughput.z) / 3.0;

        let roulette = rnd(t);
        if (roulette > prob) {
            hit.has_hit = true;
            return vec3f(0.0, 0.0, 0.0);
        }

        let cos2_t = 1 - ((hit.n1n2 * hit.n1n2) * (1 - (cos_i * cos_i)));

        R = fresnel_R(cos_i, sqrt(cos2_t), n2, n1);
        if (cos2_t < 0.0) {
            R = 1;
        }
    }

    let cos2_t = 1 - ((hit.n1n2 * hit.n1n2) * (1 - (cos_i * cos_i)));

    // var R = fresnel_R(cos_i, sqrt(cos2_t), hit.n1n2);
    if (cos2_t < 0.0) {
        R = 1;
    }

    let roulette = rnd(t);
    if (roulette < R) {
        return mirror(r, hit);
    }

    let mr = hit.n1n2 * (cos_i * hit.normal + r.direction) - hit.normal * sqrt(cos2_t);
    r.direction = mr;
    r.origin = hit.position;
    r.tmin = 0.01;
    r.tmax = 10000.0;
    hit.has_hit = false;
    hit.emit = true;

    return vec3f(0.0, 0.0, 0.0);
}

fn refractive(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    var cos_i = dot(r.direction, hit.normal);
    if (cos_i < 0.0) {
        //ray from outside the sphere
        hit.n1n2 = 1.0 / 1.5;
        cos_i = - cos_i;
    }
    else {
        //ray from inside the sphere
        hit.n1n2 = 1.5 / 1.0;
        hit.normal = - hit.normal;
    }
    let cos2_t = 1 - ((hit.n1n2 * hit.n1n2) * (1 - (cos_i * cos_i)));
    // if (cos2_t < 0.0) {
    //     return mirror(r, hit);
    // }

    let mr = hit.n1n2 * (cos_i * hit.normal + r.direction) - hit.normal * sqrt(cos2_t);
    r.direction = mr;
    r.origin = hit.position;
    r.tmin = 0.01;
    r.tmax = 10000.0;
    hit.has_hit = false;
    return vec3f(0.0, 0.0, 0.0);
}

fn mirror(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    //⋯ not for now
    let mr = reflect(r.direction, hit.normal);
    r.direction = mr;
    r.origin = hit.position;
    r.tmin = 0.01;
    r.tmax = 10000.0;
    hit.has_hit = false;
    hit.emit = true;
    return vec3f(0.0, 0.0, 0.0);
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, t: ptr<function, u32>) -> vec3f {
    switch hit.shader {
        case 1 {
            return lambertian(r, hit, t);
        }
        case 2 {
            return phong(r, hit);
        }
        case 3 {
            return mirror(r, hit);
        }
        case 4 {
            return refractive(r, hit);
        }
        case 5 {
            return glossy(r, hit);
        }
        case 6 {
            return fresnel(r, hit, t);
        }
        case default {
            return hit.color;
        }
    }
}

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex: u32) -> VSOut {
    const pos = array<vec2f, 4>(vec2f(- 1.0, 1.0), vec2f(- 1.0, - 1.0), vec2f(1.0, 1.0), vec2f(1.0, - 1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}

// fn concentricSampleDisk(u: vec2f) -> vec2f {
//     // Map uniform random numbers to [-1,1]^2
//     let uOffset = 2.0 * u - vec2f(1.0, 1.0);

//     if (uOffset.x == 0.0 && uOffset.y == 0.0) {
//         return vec2f(0.0, 0.0);
//     }

//     var r: f32;
//     var theta: f32;

//     if (abs(uOffset.x) > abs(uOffset.y)) {
//         r = uOffset.x;
//         theta = (3.141592653589793 / 4.0) * (uOffset.y / uOffset.x);
//     }
//     else {
//         r = uOffset.y;
//         theta = (3.141592653589793 / 2.0) - (3.141592653589793 / 4.0) * (uOffset.x / uOffset.y);
//     }

//     return vec2f(r * cos(theta), r * sin(theta));
// }

fn get_camera_ray(ipcoords: vec2f, t: ptr<function, u32>) -> Ray {
    var qc = vec3f(ipcoords.x, ipcoords.y, uniforms.cam_const);
    var q = mat3x3f(uniforms.b1, uniforms.b2, uniforms.v) * qc;
    var w = normalize(q);

    let focal_point = uniforms.eye + w * uniforms.focal_distance; // ft in article

    let theta = rnd(t) * 2.0 * 3.141592653589793; // random value between [0, 2Pi]

    let lens_sample = uniforms.aperture_radius * vec2f(rnd(t) * cos(theta), rnd(t) * sin(theta));
    // (pLens) - get random point on lens

    let lens_origin = uniforms.eye + uniforms.b1 * lens_sample.x + uniforms.b2 * lens_sample.y;
    //ray.o

    let new_direction = normalize(focal_point - lens_origin);
    //ray.d

    return Ray(lens_origin, new_direction, 0.001, 10000.0);
}

// PRNG xorshift seed generator by NVIDIA
fn tea(val0: u32, val1: u32) -> u32 {
    const N = 16u; // User specified number of iterations
    var v0 = val0;
    var v1 = val1;
    var s0 = 0u;
    for (var n = 0u; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

// Generate random unsigned int in [0, 2^31)
fn mcg31(prev: ptr<function, u32>) -> u32 {
    const LCG_A = 1977654935u; // Multiplier from Hui-Ching Tang [EJOR 2007]
    * prev = (LCG_A * (*prev)) & 0x7FFFFFFF;
    return * prev;
}

// Generate random float in [0, 1)
fn rnd(prev: ptr<function, u32>) -> f32 {
    return f32(mcg31(prev)) / f32(0x80000000);
}

@fragment
fn main_fs(@builtin(position) fragcoord: vec4f, @location(0) coords: vec2f) -> FSOut {
    let launch_idx = u32(fragcoord.y) * uniforms.width + u32(fragcoord.x);
    var t = tea(launch_idx, uniforms.frame);
    let prog_jitter = vec2f(rnd(&t), rnd(&t)) / (f32(uniforms.height) * sqrt(f32(uniforms.jitterindex)));

    // Progressive ray tracing
    var bgcolor = vec4f(0.1, 0.3, 0.6, 1.0);
    if (uniforms.blue_background == 0u) {
        bgcolor = vec4f(0.0, 0.0, 0.0, 1.0);
    }
    const max_depth = 10;
    let uv = vec2f(coords.x * uniforms.aspect * 0.5f, coords.y * 0.5f);

    var result = vec3f(0.0);

    // Add progressive jitter to the UV coordinates
    let rand1 = rnd(&t);
    let rand2 = rnd(&t);

    let jittered_uv = uv + prog_jitter;
    //var r = get_camera_ray(jittered_uv);
    var r = get_camera_ray(jittered_uv, & t);
    var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 0u, 0f, 0f, coords, 0u, true, vec3f(1.0), vec3f(0.0));

    for (var i = 0; i < max_depth; i++) {
        if (intersect_scene(&r, & hit)) {
            result += hit.throughput * shade(&r, & hit, & t);
            //* hit.throughput - first hit
        }
        else {
            result += hit.throughput * bgcolor.rgb;
            //this aswell
            break;
        }
        if (hit.has_hit) {
            break;
        }
    }

    let curr_sum = textureLoad(renderTexture, vec2u(fragcoord.xy), 0).rgb * f32(uniforms.frame);
    var accum_color = (result + curr_sum) / f32(uniforms.frame + 1u);
    var fsOut: FSOut;
    fsOut.frame = vec4f(pow(accum_color, vec3f(1.0 / uniforms.gamma)), bgcolor.a);
    fsOut.accum = vec4f(accum_color, 1.0);
    return fsOut;
}

// @fragment
// fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f {
//     const bgcolor = vec4f(0.1, 0.3, 0.6, 1.0);
//     const max_depth = 10;
//     let uv = vec2f(coords.x * uniforms.aspect * 0.5f, coords.y * 0.5f);

//     var result = vec3f(0.0);

//     //let uv = 0.5 * vec2f(coords.x * uniforms.aspect + jitter[uniforms.jitterindex].x, coords.y + jitter[uniforms.jitterindex].y);
//     for (var j = 0u; j < uniforms.jitterindex; j = j + 1u) {
//         var r = get_camera_ray(uv + jitter[j]);
//         var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 0u, 0f, 0f, coords, 0u);
//         for (var i = 0; i < max_depth; i++) {
//             if (intersect_scene(&r, & hit)) {
//                 result += shade(&r, & hit);
//             }
//             else {
//                 result += bgcolor.rgb;
//                 break;
//             }
//             if (hit.has_hit) {
//                 break;
//             }
//         }
//     }

//     return vec4f(pow(result / f32(uniforms.jitterindex), vec3f(1.0 / uniforms.gamma)), bgcolor.a);
// }

fn intersect_trimesh(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    var branch_lvl = 0u;
    var near_node = 0u;
    var far_node = 0u;
    var t = 0.0f;
    var node = 0u;

    for (var i = 0u; i <= MAX_LEVEL; i++) {
        let tree_node = bspTree[node];
        let node_axis_leaf = tree_node.x & 3u;
        if (node_axis_leaf == BSP_LEAF) {
            // A leaf was found ⋮
            let node_count = tree_node.x >> 2u;
            let node_id = tree_node.y;
            var found = false;
            for (var j = 0u; j < node_count; j++) {
                let obj_idx = treeIds[node_id + j];
                if (intersect_triangle(*r, hit, obj_idx)) {
                    r.tmax = hit.dist;
                    found = true;
                }
            }

            if (found) {
                return true;
            }
            else if (branch_lvl == 0u) {
                return false;
            }
            else {
                branch_lvl--;
                i = branch_node[branch_lvl].x;
                node = branch_node[branch_lvl].y;
                r.tmin = branch_ray[branch_lvl].x;
                r.tmax = branch_ray[branch_lvl].y;
                continue;
            }
        }

        let axis_direction = r.direction[node_axis_leaf];
        let axis_origin = r.origin[node_axis_leaf];
        if (axis_direction >= 0.0f) {
            near_node = tree_node.z;
            // left
            far_node = tree_node.w;
            // right
        }
        else {
            near_node = tree_node.w;
            // right
            far_node = tree_node.z;
            // left
        }

        let node_plane = bspPlanes[node];
        let denom = select(axis_direction, 1.0e-8f, abs(axis_direction) < 1.0e-8f);
        t = (node_plane - axis_origin) / denom;
        if (t > r.tmax) {
            node = near_node;
        }
        else if (t < r.tmin) {
            node = far_node;
        }
        else {
            branch_node[branch_lvl].x = i;
            branch_node[branch_lvl].y = far_node;
            branch_ray[branch_lvl].x = t;
            branch_ray[branch_lvl].y = r.tmax;
            branch_lvl++;
            r.tmax = t;
            node = near_node;
        }
    }
    return false;
}

fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    const triangle_color = vec3f(0.9);

    const sphere_center = vec3f(420, 90, 370.0);
    const sphere_radius = 90;
    const sphere_refractive_index = 1.5;
    //hardcoded in fresnel
    const sphere_shininess = 42.0;
    const sphere_color = vec3f(0.0, 0.0, 0.0);

    if (intersect_sphere(*r, hit, sphere_center, sphere_radius)) {
        r.tmax = hit.dist;
        hit.color = sphere_color;
        hit.specular = vec3f(0.1, 0.1, 0.1);
        hit.sphere_shininess = 42;
        hit.emission = hit.color * 0.1;
        hit.diffuse = hit.color * 0.9;
        hit.shader = 3u;
        //hit.shader = 5u;
    }

    const sphere_center2 = vec3f(130.0, 90.0, 250.0);
    const sphere_radius2 = 90;
    const sphere_refractive_index2 = 1.5;
    const sphere_shininess2 = 42.0;
    const sphere_color2 = vec3f(0.0, 0.0, 0.0);

    if (intersect_sphere(*r, hit, sphere_center2, sphere_radius2)) {
        r.tmax = hit.dist;
        hit.color = sphere_color2;
        hit.specular = vec3f(0.1, 0.1, 0.1);
        hit.sphere_shininess = 42;
        hit.emission = hit.color * 0.1;
        hit.diffuse = hit.color * 0.9;
        hit.shader = 6u;
        hit.sigma_t = vec3f(0.0, 0.1, 0.1);
        //inverted rgb
        //hit.shader = 5u;
    }

    if (intersect_min_max(r)) {
        //for (var i = 0u; i < arrayLength(&meshFaces); i++) {
        //if (intersect_triangle(*r, hit, i)) {
        if (intersect_trimesh(r, hit)) {
            r.tmax = hit.dist;
            hit.color = materials[meshFaces[hit.face_idx].w].diffuse + materials[meshFaces[hit.face_idx].w].emission;
            hit.diffuse = materials[meshFaces[hit.face_idx].w].diffuse;
            hit.specular = vec3f(0, 0, 0);
            hit.emission = materials[meshFaces[hit.face_idx].w].emission;
            hit.shader = 1u;
        }
        //}
    }
    else {
        //hit.color = vec3f(1, 0, 0);
        hit.shader = 0;
        hit.has_hit = true;
    }
    return hit.has_hit;
}

fn intersect_min_max(r: ptr<function, Ray>) -> bool {
    let p1 = (aabb.min - r.origin) / r.direction;
    let p2 = (aabb.max - r.origin) / r.direction;
    let pmin = min(p1, p2);
    let pmax = max(p1, p2);
    let box_tmin = max(pmin.x, max(pmin.y, pmin.z)) - 1.0e-2f;
    let box_tmax = min(pmax.x, min(pmax.y, pmax.z)) + 1.0e-2f;
    if (box_tmin > box_tmax || box_tmin > r.tmax || box_tmax < r.tmin) {
        return false;
    }
    r.tmin = max(box_tmin, r.tmin);
    r.tmax = min(box_tmax, r.tmax);
    return true;
}
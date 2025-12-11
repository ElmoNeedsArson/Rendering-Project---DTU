"use strict";
window.onload = function () { main(); }

import { cameraPresets, presetMetaElements } from './scripts/presets.js';

async function main() {
    const gpu = navigator.gpu;
    const adapter = await gpu.requestAdapter();
    const canTimestamp = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({ requiredFeatures: [...(canTimestamp ? ['timestamp-query'] : []),], });
    const canvas = document.getElementById('my-canvas');

    const preset1a = document.getElementById("preset1a");
    const preset1b = document.getElementById("preset1b");
    const preset2a = document.getElementById("preset2a");
    const preset2b = document.getElementById("preset2b");
    const preset3a = document.getElementById("preset3a");
    const preset3b = document.getElementById("preset3b");
    const preset4a = document.getElementById("preset4a");
    const preset4b = document.getElementById("preset4b");
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    const gamma = document.getElementById("gamma");
    const checkboxProgressiveRendering = document.getElementById("checkbox");
    const blueBackground = document.getElementById("blueBackground");
    const infoElem = document.getElementById('infoEl');
    const lensRadius = document.getElementById("lensRadius");
    const focalDistance = document.getElementById("focalDistance");
    const gammaValueLabel = document.getElementById("gammaValue");
    const lensRadiusValueLabel = document.getElementById("lensRadiusValue");
    const focalDistanceValueLabel = document.getElementById("focalDistanceValue");
    const fl_val_label = document.getElementById("fl_val");
    const allPresetCards = document.querySelectorAll('.preset-card');

    let obj_filename = "../models/boat_resized+cornell_pos.obj";

    let mouseMoving = false;
    let frame = 0;
    const aspect = canvas.width / canvas.height;
    const cam_const = 0.5;
    let eye = vec3(70, 100, 130);
    let lookat = vec3(900, 100, -80);
    let up = vec3(0.0, 1.0, 0.0);
    let yaw = 0;        // horizontal rotation in radians
    let pitch = 0;      // vertical rotation in radians

    canvas.addEventListener("click", () => {
        canvas.requestPointerLock();
    });

    canvas.addEventListener("mousemove", (e) => {
        if (document.pointerLockElement === canvas) {
            mouseMoving = true;

            const sens = 0.0025;
            yaw += e.movementX * sens;
            pitch -= e.movementY * sens;

            // Clamp pitch to avoid flipping over at the poles.
            const maxPitch = Math.PI / 2 - 0.01;
            if (pitch > maxPitch) pitch = maxPitch;
            if (pitch < -maxPitch) pitch = -maxPitch;
        }
    });

    canvas.addEventListener("contextmenu", (ev) => {
        ev.preventDefault(); // stop the default right click menu
    });

    async function load() {
        const obj = await readOBJFile(obj_filename, 1, true);

        const timingHelper = new TimingHelper(device);
        let gpuTime = 0;
        let buffers = {};
        buffers = build_bsp_tree(obj, device, buffers);

        blueBackground.addEventListener('change', () => {
            device.queue.writeBuffer(uniformBuffer, 60, new Uint32Array([blueBackground.checked ? 1 : 0]));
            kickstartAnimate(frame);
            update();
        })

        checkboxProgressiveRendering.addEventListener('change', () => {
            animate();
        })

        lensRadius.addEventListener('input', () => {
            lensRadiusValueLabel.textContent = Number(lensRadius.value).toFixed(1);
            device.queue.writeBuffer(uniformBuffer, 80, flatten([lensRadius.value]));
            kickstartAnimate(frame);
            update();
        })

        focalDistance.addEventListener('input', () => {
            focalDistanceValueLabel.textContent = Number(focalDistance.value).toFixed(1);
            let fd = parseFloat(focalDistance.value);
            fl_val_label.textContent = 'Focal length: ' + (fd * cam_const / (fd + cam_const)).toFixed(2);
            device.queue.writeBuffer(uniformBuffer, 84, flatten([focalDistance.value]));
            kickstartAnimate(frame);
            update();
        })

        gamma.addEventListener('input', () => {
            gammaValueLabel.textContent = Number(gamma.value).toFixed(1);
            device.queue.writeBuffer(uniformBuffer, 76, new Float32Array([gamma.value]));
            update();
        })

        // Attach one listener per button, but delegate logic
        preset1a.addEventListener('click', () => applyCameraPreset('preset1a'));
        preset1b.addEventListener('click', () => applyCameraPreset('preset1b'));
        preset2a.addEventListener('click', () => applyCameraPreset('preset2a'));
        preset2b.addEventListener('click', () => applyCameraPreset('preset2b'));
        preset3a.addEventListener('click', () => applyCameraPreset('preset3a'));
        preset3b.addEventListener('click', () => applyCameraPreset('preset3b'));
        preset4a.addEventListener('click', () => applyCameraPreset('preset4a'));
        preset4b.addEventListener('click', () => applyCameraPreset('preset4b'));

        function updateCamera() {
            frame = 0;

            const cy = Math.cos(pitch);
            const sy = Math.sin(pitch);
            const cx = Math.cos(yaw);
            const sx = Math.sin(yaw);

            // Direction vector from yaw/pitch
            let v = vec3(cx * cy, sy, sx * cy);
            v = normalize(v);
            lookat = add(eye, v);

            let b1 = normalize(cross(v, up));
            let b2 = cross(b1, v);

            device.queue.writeBuffer(uniformBuffer, 16, flatten([...eye]));
            device.queue.writeBuffer(uniformBuffer, 32, flatten([...b1]));
            device.queue.writeBuffer(uniformBuffer, 48, flatten([...b2]));
            device.queue.writeBuffer(uniformBuffer, 64, flatten([...v]));
        }

        // Initialize preset card meta text from cameraPresets
        Object.keys(cameraPresets).forEach(id => {
            const preset = cameraPresets[id];
            const el = presetMetaElements[id];
            if (preset && el && preset.focalDistance !== undefined) {
                el.textContent = `Focal distance: ${preset.focalDistance}`;
            }
        });

        function kickstartAnimate(oldFrameCount) {
            console.log("frame: " + oldFrameCount);
            frame = 0;
            if (oldFrameCount >= 500) {
                animate(); //kickstart animate again haha
            }
        }

        // Shared handler for camera preset buttons
        function applyCameraPreset(presetId) {
            const preset = cameraPresets[presetId];
            if (!preset) return;

            eye = preset.eye;
            lookat = preset.lookat;

            // When we apply a preset, we simply set yaw/pitch to the stored values so that the camera snaps cleanly to it.
            yaw = preset.yaw;
            pitch = preset.pitch;

            // Update depth-of-field controls from preset
            lensRadius.value = preset.lensRadius;
            lensRadiusValueLabel.textContent = Number(preset.lensRadius).toFixed(1);
            focalDistance.value = preset.focalDistance;
            focalDistanceValueLabel.textContent = Number(preset.focalDistance).toFixed(1);
            device.queue.writeBuffer(uniformBuffer, 84, flatten([preset.focalDistance]));
            let fd = parseFloat(focalDistance.value);
            fl_val_label.textContent = 'Focal length: ' + (fd * cam_const / (fd + cam_const)).toFixed(2);

            let oldFrameCount = frame;

            updateCamera();
            update();
            kickstartAnimate(oldFrameCount)

            // Update active preset highlight
            allPresetCards.forEach(card => card.classList.remove('active'));
            const activeEl = document.getElementById(presetId);
            if (activeEl) {
                activeEl.classList.add('active');
            }
        }

        const wgslfile = document.getElementById('wgsl').src;
        const wgslcode = await fetch(wgslfile, { cache: "reload" }).then(r => r.text());
        const wgsl = device.createShaderModule({
            code: wgslcode
        });

        let textures = new Object();
        textures.width = canvas.width;
        textures.height = canvas.height;
        textures.renderSrc = device.createTexture({
            size: [canvas.width, canvas.height],
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
            format: 'rgba32float',
        });
        textures.renderDst = device.createTexture({
            size: [canvas.width, canvas.height],
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            format: 'rgba32float',
        });

        const pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: wgsl,
                entryPoint: 'main_vs',
                buffers: [],
            },
            fragment: {
                module: wgsl,
                entryPoint: 'main_fs',
                targets: [{ format: canvasFormat }, { format: "rgba32float" }],
            },
            primitive: { topology: 'triangle-strip', },
        });

        let bytelength = 8 * sizeof['vec4'];
        let uniforms = new ArrayBuffer(bytelength);
        const uniformBuffer = device.createBuffer({
            size: uniforms.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        let mat_bytelength = obj.materials.length * 2 * sizeof['vec4'];
        var materials = new ArrayBuffer(mat_bytelength);
        const materialBuffer = device.createBuffer({
            size: mat_bytelength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        });
        for (var i = 0; i < obj.materials.length; i++) {
            const mat = obj.materials[i];

            const e = mat.emission || { r: 0, g: 0, b: 0, a: 1 };
            const c = mat.color || { r: 0.8, g: 0.8, b: 0.8, a: 1 };

            const emission = vec4(e.r, e.g, e.b, e.a);
            const color = vec4(c.r, c.g, c.b, c.a);
            new Float32Array(materials, i * 2 * sizeof['vec4'], 8).set([...emission, ...color]);
        }
        device.queue.writeBuffer(materialBuffer, 0, materials);

        const lightIndicesBuffer = device.createBuffer({
            size: obj.light_indices.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
        });
        device.queue.writeBuffer(lightIndicesBuffer, 0, obj.light_indices);

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: textures.renderDst.createView() },
                { binding: 3, resource: { buffer: buffers.attribs } },
                { binding: 4, resource: { buffer: buffers.indices } },
                { binding: 6, resource: { buffer: materialBuffer } },
                { binding: 7, resource: { buffer: lightIndicesBuffer } },
                { binding: 8, resource: { buffer: buffers.aabb } },
                { binding: 9, resource: { buffer: buffers.treeIds } },
                { binding: 10, resource: { buffer: buffers.bspTree } },
                { binding: 11, resource: { buffer: buffers.bspPlanes } },
            ],
        });

        let v = normalize(subtract(lookat, eye));
        let b1 = normalize(cross(v, up));
        let b2 = cross(b1, v);

        new Float32Array(uniforms, 0, 32).set([
            aspect, cam_const, 0, 0, // 16
            ...eye, 0, // 32
            ...b1, 0, // 48
            ...b2, 0, // 64
            ...v, gamma.value, // 80
            lensRadius.value, focalDistance.value, 0, 0 // 96
        ]);
        new Uint32Array(uniforms, 8, 3).set([1]);
        new Uint32Array(uniforms, 12, 1).set([canvas.width]);
        new Uint32Array(uniforms, 28, 1).set([canvas.height]);
        new Uint32Array(uniforms, 44, 1).set([frame]);
        new Uint32Array(uniforms, 60, 1).set([blueBackground.checked ? 1 : 0]);
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);

        applyCameraPreset('preset1a');

        function update() {
            device.queue.writeBuffer(uniformBuffer, 44, new Uint32Array([frame]));
            const encoder = device.createCommandEncoder();
            const pass = timingHelper.beginRenderPass(encoder, {
                colorAttachments: [{
                    view: context.getCurrentTexture().createView(),
                    loadOp: "clear", storeOp: "store",
                }, {
                    view: textures.renderSrc.createView(),
                    loadOp: "load", storeOp: "store"
                }]
            });

            pass.setBindGroup(0, bindGroup);
            pass.setPipeline(pipeline);
            pass.draw(4);
            pass.end();

            encoder.copyTextureToTexture(
                { texture: textures.renderSrc },
                { texture: textures.renderDst },
                [textures.width, textures.height]
            );

            device.queue.submit([encoder.finish()]);
            timingHelper.getResult().then(time => { gpuTime = time / 1000; });
            let gpu = `${canTimestamp && gpuTime > 0 ? `${(gpuTime / 1000).toFixed(1)} ms` : 'N/A'}`
            infoElem.textContent = `\ gpu: ${gpu} | Frame: ${frame}`
            frame++;
        }

        update()

        function animate() { // Perform one progressive rendering step
            if (checkboxProgressiveRendering.checked && frame < 500) {
                if (mouseMoving) {
                    updateCamera();
                    mouseMoving = false;
                }
                update();
                requestAnimationFrame(animate);
            }
        }
        animate();
    }
    load();
}
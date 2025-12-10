export const cameraPresets = {
    preset1a: {
        eye: vec3(355, 100, 115),
        lookat: vec3(900, 100, -80),
        yaw: 2.2,
        pitch: 0.05,
        lensRadius: 0.7,
        focalDistance: 116.2,
    },
    preset1b: {
        eye: vec3(355, 100, 115),
        lookat: vec3(900, 100, -80),
        yaw: 2.2,
        pitch: 0.05,
        lensRadius: 0.7,
        focalDistance: 35,
    },

    preset2a: {
        eye: vec3(277, 275, -570),
        lookat: vec3(277, 275, 0.0),
        yaw: 1.59,
        pitch: 0,
        lensRadius: 0.7,
        focalDistance: 50,
    },
    preset2b: {
        eye: vec3(277, 275, -570),
        lookat: vec3(277, 275, 0.0),
        yaw: 1.59,
        pitch: 0,
        lensRadius: 0.7,
        focalDistance: 400,
    },

    preset3a: {
        eye: vec3(70, 100, 130),
        lookat: vec3(900, 100, -80),
        yaw: 0,
        pitch: 0,
        lensRadius: 0.7,
        focalDistance: 100,
    },
    preset3b: {
        eye: vec3(70, 100, 130),
        lookat: vec3(900, 100, -80),
        yaw: 0,
        pitch: 0,
        lensRadius: 0.7,
        focalDistance: 20,
    },

    preset4a: {
        eye: vec3(355, 100, 115),
        lookat: vec3(900, 100, -80),
        yaw: 1.59,
        pitch: 0,
        lensRadius: 0.7,
        focalDistance: 19.2,
    },
    preset4b: {
        eye: vec3(355, 100, 115),
        lookat: vec3(900, 100, -80),
        yaw: 1.59,
        pitch: 0,
        lensRadius: 0.7,
        focalDistance: 100,
    },
};

export const presetMetaElements = {
    preset1a: document.getElementById("preset1aMeta"),
    preset1b: document.getElementById("preset1bMeta"),
    preset2a: document.getElementById("preset2aMeta"),
    preset2b: document.getElementById("preset2bMeta"),
    preset3a: document.getElementById("preset3aMeta"),
    preset3b: document.getElementById("preset3bMeta"),
    preset4a: document.getElementById("preset4aMeta"),
    preset4b: document.getElementById("preset4bMeta"),
};
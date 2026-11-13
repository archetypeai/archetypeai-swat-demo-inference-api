import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { existsSync, readFileSync } from 'fs';

// The project's .env wins over ATAI_* variables exported in your shell. Vite
// normally lets the shell win, so a shell profile pointing at another
// deployment (or the free trial) would silently redirect every API call.
if (existsSync('.env')) {
	for (const line of readFileSync('.env', 'utf-8').split('\n')) {
		const m = line.match(/^(ATAI_[A-Z0-9_]*)=(.*)$/);
		if (m) process.env[m[1]] = m[2].trim();
	}
}

// Vite handles Ctrl-C itself and exits before a signal handler in the app could
// finish, so cancel the server's Omega session as part of Vite's own shutdown,
// which waits for plugins to close. See getOmegaSession() in
// src/lib/server/newton.js.
const cancelOmegaSession = {
	name: 'cancel-omega-session',
	async closeBundle() {
		await globalThis.__swatOmegaSession?.close();
	}
};

export default defineConfig({ plugins: [tailwindcss(), sveltekit(), cancelOmegaSession] });

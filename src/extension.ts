import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';

let pythonProcess: ChildProcess | null = null;
let requestId = 0;
const pendingRequests = new Map<string, { resolve: (value: { result: string; expression: string }) => void; reject: (reason: Error) => void }>();
let stdoutBuffer = '';

function sendRequest(expression: string): Promise<{ result: string; expression: string }> {
	return new Promise((resolve, reject) => {
		if (!pythonProcess || !pythonProcess.stdin) {
			reject(new Error('Python process is not running'));
			return;
		}
		const id = String(++requestId);
		pendingRequests.set(id, { resolve, reject });
		const message = JSON.stringify({ id, expression }) + '\n';
		pythonProcess.stdin.write(message);
	});
}

function handleStdoutData(data: string) {
	stdoutBuffer += data;
	const lines = stdoutBuffer.split('\n');
	stdoutBuffer = lines.pop() || '';
	for (const line of lines) {
		if (!line.trim()) {
			continue;
		}
		try {
			const parsed = JSON.parse(line);
			const pending = pendingRequests.get(parsed.id);
			if (pending) {
				pendingRequests.delete(parsed.id);
				if (parsed.error) {
					pending.reject(new Error(parsed.error));
				} else {
					pending.resolve({ result: parsed.result, expression: parsed.expression });
				}
			}
		} catch {
			console.error('Failed to parse Python output:', line);
		}
	}
}

function spawnPython(scriptPath: string): Promise<void> {
	return new Promise((resolve, reject) => {
		const trySpawn = (command: string, fallback?: string) => {
			const proc = spawn(command, [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] });

			let settled = false;
			let readyBuffer = '';

			proc.stdout!.setEncoding('utf-8');
			proc.stderr!.setEncoding('utf-8');

			const onStdoutData = (data: string) => {
				readyBuffer += data;
				const lines = readyBuffer.split('\n');
				readyBuffer = lines.pop() || '';
				for (const line of lines) {
					if (!line.trim()) {
						continue;
					}
					try {
						const parsed = JSON.parse(line);
						if (parsed.status === 'ready') {
							settled = true;
							pythonProcess = proc;
							// Switch to the normal handler for subsequent data
							proc.stdout!.removeListener('data', onStdoutData);
							// Process any remaining buffered data with the normal handler
							stdoutBuffer = readyBuffer;
							proc.stdout!.on('data', handleStdoutData);
							resolve();
							return;
						}
					} catch {
						// Not JSON, ignore during startup
					}
				}
			};

			proc.stdout!.on('data', onStdoutData);

			proc.stderr!.on('data', (data: string) => {
				console.error('Python stderr:', data);
			});

			proc.on('error', (err) => {
				if (!settled) {
					settled = true;
					if (fallback) {
						trySpawn(fallback);
					} else {
						reject(new Error(`Failed to start Python: ${err.message}`));
					}
				}
			});

			proc.on('close', (code) => {
				if (!settled) {
					settled = true;
					if (fallback) {
						trySpawn(fallback);
					} else {
						reject(new Error(`Python process exited with code ${code}`));
					}
				} else if (proc === pythonProcess) {
					// Only clear if this is the active process (not a failed primary spawn)
					pythonProcess = null;
					for (const [id, pending] of pendingRequests) {
						pending.reject(new Error('Python process exited'));
						pendingRequests.delete(id);
					}
				}
			});
		};

		trySpawn('python3', 'python');
	});
}

export async function activate(context: vscode.ExtensionContext) {
	const scriptPath = path.join(context.extensionPath, 'server.py');

	try {
		await spawnPython(scriptPath);
	} catch (error) {
		vscode.window.showErrorMessage(`SuggestTeX: Failed to start Python process. ${error}`);
		return;
	}

	const provider1 = vscode.languages.registerCompletionItemProvider('latex', {
		provideCompletionItems(document: vscode.TextDocument, position: vscode.Position, token: vscode.CancellationToken, context: vscode.CompletionContext) {
			const commitCharacterCompletion = new vscode.CompletionItem(' =');
			commitCharacterCompletion.commitCharacters = ['='];
			commitCharacterCompletion.documentation = new vscode.MarkdownString('Press `=` to get `console.`');
			return [commitCharacterCompletion];
		}
	});

	const provider2 = vscode.languages.registerCompletionItemProvider(
		'latex',
		{
			async provideCompletionItems(document: vscode.TextDocument, position: vscode.Position) {
				const linePrefix = document.lineAt(position).text.slice(0, position.character);
				const linePrefix2 = document.lineAt(position).text.slice(0, position.character - 2);
				console.log(linePrefix);
				if (!linePrefix.endsWith(' =')) {
					console.log('not end with \' =\'');
					return undefined;
				}

				const replacedString = linePrefix2.replace(/\\SI{([0-9.}]+)}{[^}]*}/g, '$1');
				console.log(replacedString);

				try {
					const response = await sendRequest(replacedString);
					console.log('Result:', response.result);
					console.log('Expression:', response.expression);

					const resultText = new vscode.CompletionItem(" " + response.result, vscode.CompletionItemKind.Method);
					resultText.documentation = response.expression;

					return [resultText];
				} catch (error) {
					console.error('Error:', error);
					return [];
				}
			}
		},
		'=' // triggered whenever a '.' is being typed
	);

	context.subscriptions.push(provider1, provider2);
}

export function deactivate() {
	for (const [, pending] of pendingRequests) {
		pending.reject(new Error('Extension deactivating'));
	}
	pendingRequests.clear();
	if (pythonProcess) {
		pythonProcess.kill();
		pythonProcess = null;
	}
}

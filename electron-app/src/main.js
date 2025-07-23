const { app, BrowserWindow, Menu, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const express = require('express');
const cors = require('cors');
const fs = require('fs');

// Keep a global reference of the window object
let mainWindow;
let pythonProcess = null;
let expressApp = null;
let server = null;

// Development mode check
const isDev = process.argv.includes('--dev');

// Paths for bundled resources
const getResourcePath = (relativePath) => {
  if (isDev) {
    return path.join(__dirname, '../../', relativePath);
  }
  return path.join(process.resourcesPath, relativePath);
};

const getPythonPath = () => {
  if (isDev) {
    return 'python'; // Use system Python in dev mode
  }
  // Use bundled Python environment
  const pythonEnvPath = path.join(process.resourcesPath, 'python-env');
  return path.join(pythonEnvPath, 'bin', 'python');
};

const getBackendPath = () => {
  if (isDev) {
    return path.join(__dirname, '../../main.py');
  }
  return path.join(process.resourcesPath, 'backend', 'main.py');
};

// Start Python backend server
async function startPythonBackend() {
  return new Promise((resolve, reject) => {
    const pythonPath = getPythonPath();
    const backendPath = getBackendPath();
    
    console.log('Starting Python backend...');
    console.log('Python path:', pythonPath);
    console.log('Backend path:', backendPath);
    
    // Set environment variables
    const env = { ...process.env };
    if (!isDev) {
      // Add bundled Python environment to PATH
      const pythonEnvPath = path.join(process.resourcesPath, 'python-env');
      env.PATH = `${path.join(pythonEnvPath, 'bin')}:${env.PATH}`;
      env.PYTHONPATH = path.join(pythonEnvPath, 'lib', 'python3.13', 'site-packages');
    }
    
    pythonProcess = spawn(pythonPath, ['-c', `
import sys
import os
import uvicorn

# Add backend directory to Python path
backend_dir = "${path.dirname(backendPath)}"
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Change to backend directory
os.chdir("${isDev ? path.dirname(backendPath) : path.dirname(backendPath)}")

# Import and run the app
sys.path.insert(0, '.')
import main
uvicorn.run(main.app, host='127.0.0.1', port=8000, log_level='info')
`], {
      env: env,
      cwd: isDev ? path.dirname(backendPath) : path.dirname(backendPath)
    });

    pythonProcess.stdout.on('data', (data) => {
      console.log('Backend:', data.toString());
      if (data.toString().includes('Uvicorn running')) {
        console.log('Python backend started successfully');
        resolve();
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error('Backend Error:', data.toString());
      if (data.toString().includes('Address already in use')) {
        console.log('Backend already running, continuing...');
        resolve();
      }
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python backend exited with code ${code}`);
      if (code !== 0) {
        reject(new Error(`Python backend failed with code ${code}`));
      }
    });

    pythonProcess.on('error', (error) => {
      console.error('Failed to start Python backend:', error);
      reject(error);
    });

    // Timeout after 30 seconds
    setTimeout(() => {
      console.log('Backend startup timeout, continuing anyway...');
      resolve();
    }, 30000);
  });
}

// Start Express server for frontend
function startFrontendServer() {
  return new Promise((resolve) => {
    expressApp = express();
    expressApp.use(cors());
    
    // Serve static files from frontend build
    const frontendPath = isDev 
      ? path.join(__dirname, '../../playstore-reviews-frontend/out')
      : path.join(process.resourcesPath, 'frontend', 'out');
    
    console.log('Serving frontend from:', frontendPath);
    
    if (fs.existsSync(frontendPath)) {
      expressApp.use(express.static(frontendPath));
      
      // Handle SPA routing
      expressApp.get('*', (req, res) => {
        res.sendFile(path.join(frontendPath, 'index.html'));
      });
    } else {
      // Fallback: serve a simple HTML page
      expressApp.get('*', (req, res) => {
        res.send(`
          <!DOCTYPE html>
          <html>
          <head>
            <title>PlayStore Review Scraper</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
              .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
              h1 { color: #333; text-align: center; }
              .status { padding: 20px; background: #e8f5e8; border-radius: 4px; margin: 20px 0; }
              .api-link { display: inline-block; padding: 10px 20px; background: #007cba; color: white; text-decoration: none; border-radius: 4px; margin: 10px 0; }
              .api-link:hover { background: #005a87; }
            </style>
          </head>
          <body>
            <div class="container">
              <h1>🎯 PlayStore Review Scraper</h1>
              <div class="status">
                <strong>✅ Application is running successfully!</strong><br>
                Backend API is available at: <code>http://localhost:8000</code>
              </div>
              <p>This is the desktop version of PlayStore Review Scraper with full AI analysis capabilities.</p>
              <a href="http://localhost:8000/docs" class="api-link" target="_blank">📚 Open API Documentation</a>
              <a href="http://localhost:8000/health" class="api-link" target="_blank">🏥 Check Health Status</a>
            </div>
          </body>
          </html>
        `);
      });
    }
    
    server = expressApp.listen(3000, '127.0.0.1', () => {
      console.log('Frontend server started on http://localhost:3000');
      resolve();
    });
  });
}

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      webSecurity: true
    },
    titleBarStyle: 'default',
    show: false // Don't show until ready
  });

  // Load the frontend
  mainWindow.loadURL('http://localhost:3000');

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Create application menu
  createMenu();
}

function createMenu() {
  const template = [
    {
      label: 'PlayStore Review Scraper',
      submenu: [
        {
          label: 'About PlayStore Review Scraper',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About',
              message: 'PlayStore Review Scraper v2.0.0',
              detail: 'AI-powered review analysis tool for Google Play Store apps.\n\nFeatures:\n• 30,000 review analysis\n• Advanced sentiment analysis\n• Topic modeling and clustering\n• Actionable insights generation'
            });
          }
        },
        { type: 'separator' },
        {
          label: 'Open API Documentation',
          click: () => {
            shell.openExternal('http://localhost:8000/docs');
          }
        },
        { type: 'separator' },
        { role: 'quit' }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectall' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forcereload' },
        { role: 'toggledevtools' },
        { type: 'separator' },
        { role: 'resetzoom' },
        { role: 'zoomin' },
        { role: 'zoomout' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'close' }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// App event handlers
app.whenReady().then(async () => {
  console.log('Electron app ready, starting services...');
  
  try {
    // Start backend and frontend servers
    await Promise.all([
      startPythonBackend(),
      startFrontendServer()
    ]);
    
    console.log('All services started, creating window...');
    createWindow();
    
  } catch (error) {
    console.error('Failed to start services:', error);
    dialog.showErrorBox('Startup Error', `Failed to start application services: ${error.message}`);
    app.quit();
  }
});

app.on('window-all-closed', () => {
  // On macOS, keep app running even when all windows are closed
  if (process.platform !== 'darwin') {
    cleanup();
    app.quit();
  }
});

app.on('activate', () => {
  // On macOS, re-create window when dock icon is clicked
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on('before-quit', () => {
  cleanup();
});

function cleanup() {
  console.log('Cleaning up processes...');
  
  if (pythonProcess) {
    pythonProcess.kill('SIGTERM');
    pythonProcess = null;
  }
  
  if (server) {
    server.close();
    server = null;
  }
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  cleanup();
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
}); 
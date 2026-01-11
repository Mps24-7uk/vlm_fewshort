# nodes/ui.py
import threading
import webbrowser
import os
import time
import socket
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn


def render_review_html(predicted_class, reason, image_url):
    """Render the main review page HTML."""
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Human Review System</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f172a;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.1) 0px, transparent 50%),
                    radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.1) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
                    radial-gradient(at 0% 100%, rgba(168, 85, 247, 0.1) 0px, transparent 50%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 30px;
                line-height: 1.6;
            }}
            
            .container {{
                background: rgba(255, 255, 255, 0.98);
                border-radius: 24px;
                box-shadow: 
                    0 0 0 1px rgba(0, 0, 0, 0.05),
                    0 20px 60px -15px rgba(0, 0, 0, 0.3),
                    0 40px 100px -20px rgba(0, 0, 0, 0.2);
                max-width: 700px;
                width: 100%;
                padding: 48px;
                animation: slideIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
                backdrop-filter: blur(10px);
            }}
            
            @keyframes slideIn {{
                from {{
                    opacity: 0;
                    transform: translateY(-20px) scale(0.95);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }}
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 36px;
                padding-bottom: 24px;
                border-bottom: 2px solid #f1f5f9;
            }}
            
            .header-icon {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 56px;
                height: 56px;
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                border-radius: 16px;
                margin-bottom: 16px;
                box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.5);
            }}
            
            .header-icon svg {{
                width: 28px;
                height: 28px;
                stroke: white;
                fill: none;
                stroke-width: 2;
            }}
            
            h2 {{
                color: #0f172a;
                font-size: 26px;
                font-weight: 700;
                margin-bottom: 8px;
                letter-spacing: -0.5px;
            }}
            
            .subtitle {{
                color: #64748b;
                font-size: 14px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .info-section {{
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 32px;
                border: 1px solid #e2e8f0;
            }}
            
            .info-item {{
                display: flex;
                align-items: flex-start;
                margin-bottom: 16px;
                gap: 12px;
            }}
            
            .info-item:last-child {{
                margin-bottom: 0;
            }}
            
            .info-label {{
                font-weight: 600;
                color: #475569;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                min-width: 110px;
                padding-top: 2px;
            }}
            
            .info-value {{
                color: #1e293b;
                font-size: 15px;
                font-weight: 500;
                flex: 1;
            }}
            
            .badge {{
                display: inline-block;
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                padding: 4px 12px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }}
            
            .image-container {{
                position: relative;
                text-align: center;
                margin: 32px 0;
                border-radius: 16px;
                overflow: hidden;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            }}
            
            img {{
                max-width: 100%;
                max-height: 450px;
                display: block;
                margin: 0 auto;
                object-fit: contain;
            }}
            
            .actions {{
                display: flex;
                gap: 16px;
                margin-top: 36px;
            }}
            
            button {{
                flex: 1;
                padding: 16px 32px;
                font-size: 15px;
                font-weight: 600;
                font-family: 'Inter', sans-serif;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                letter-spacing: 0.3px;
                position: relative;
                overflow: hidden;
            }}
            
            button::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 100%);
                opacity: 0;
                transition: opacity 0.2s;
            }}
            
            button:hover::before {{
                opacity: 1;
            }}
            
            button[value="y"] {{
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                box-shadow: 0 4px 20px -4px rgba(16, 185, 129, 0.5);
            }}
            
            button[value="y"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 30px -4px rgba(16, 185, 129, 0.6);
            }}
            
            button[value="y"]:active {{
                transform: translateY(0);
            }}
            
            button[value="n"] {{
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                color: white;
                box-shadow: 0 4px 20px -4px rgba(239, 68, 68, 0.5);
            }}
            
            button[value="n"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 30px -4px rgba(239, 68, 68, 0.6);
            }}
            
            button[value="n"]:active {{
                transform: translateY(0);
            }}
            
            .button-icon {{
                display: inline-block;
                margin-right: 8px;
                font-size: 18px;
                vertical-align: middle;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="header-icon">
                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path d="M16 7C16 9.20914 14.2091 11 12 11C9.79086 11 8 9.20914 8 7C8 4.79086 9.79086 3 12 3C14.2091 3 16 4.79086 16 7Z" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M12 14C8.13401 14 5 17.134 5 21H19C19 17.134 15.866 14 12 14Z" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <h2>Human Review Required</h2>
                <p class="subtitle">Quality Assurance Check</p>
            </div>
            
            <div class="info-section">
                <div class="info-item">
                    <span class="info-label">Prediction</span>
                    <span class="info-value"><span class="badge">{predicted_class}</span></span>
                </div>
                <div class="info-item">
                    <span class="info-label">Analysis</span>
                    <span class="info-value">{reason}</span>
                </div>
            </div>
            
            <div class="image-container">
                <img src="{image_url}" alt="Image for review" />
            </div>
            
            <form method="post">
                <div class="actions">
                    <button name="decision" value="y" type="submit">
                        <span class="button-icon">✓</span>Approve
                    </button>
                    <button name="decision" value="n" type="submit">
                        <span class="button-icon">✗</span>Reject
                    </button>
                </div>
            </form>
        </div>
        
        <script>
            // Send heartbeat to server every 2 seconds
            let heartbeatInterval = setInterval(function() {{
                fetch('/heartbeat', {{ method: 'POST' }})
                    .catch(err => console.log('Heartbeat failed'));
            }}, 2000);
            
            // Notify server when page is being closed
            window.addEventListener('beforeunload', function(e) {{
                fetch('/page-closing', {{ 
                    method: 'POST',
                    keepalive: true 
                }});
            }});
            
            // Stop heartbeat when form is submitted
            document.querySelector('form').addEventListener('submit', function() {{
                clearInterval(heartbeatInterval);
            }});
        </script>
    </body>
    </html>
    '''


def render_thank_you_html():
    """Render the thank you page HTML."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Review Submitted</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f172a;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.1) 0px, transparent 50%),
                    radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.1) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
                    radial-gradient(at 0% 100%, rgba(168, 85, 247, 0.1) 0px, transparent 50%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 30px;
            }
            
            .message-container {
                background: rgba(255, 255, 255, 0.98);
                padding: 64px 56px;
                border-radius: 24px;
                box-shadow: 
                    0 0 0 1px rgba(0, 0, 0, 0.05),
                    0 20px 60px -15px rgba(0, 0, 0, 0.3),
                    0 40px 100px -20px rgba(0, 0, 0, 0.2);
                text-align: center;
                max-width: 500px;
                animation: fadeInScale 0.6s cubic-bezier(0.16, 1, 0.3, 1);
                backdrop-filter: blur(10px);
            }
            
            @keyframes fadeInScale {
                from { 
                    opacity: 0; 
                    transform: scale(0.9);
                }
                to { 
                    opacity: 1; 
                    transform: scale(1);
                }
            }
            
            .success-icon {
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 32px;
                box-shadow: 0 10px 40px -10px rgba(16, 185, 129, 0.6);
                animation: scaleIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) 0.2s backwards;
            }
            
            @keyframes scaleIn {
                from {
                    transform: scale(0);
                    opacity: 0;
                }
                to {
                    transform: scale(1);
                    opacity: 1;
                }
            }
            
            .checkmark {
                color: white;
                font-size: 44px;
                font-weight: bold;
            }
            
            h3 {
                color: #0f172a;
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 12px;
                letter-spacing: -0.5px;
            }
            
            p {
                color: #64748b;
                font-size: 15px;
                font-weight: 500;
                line-height: 1.6;
            }
        </style>
    </head>
    <body>
        <div class="message-container">
            <div class="success-icon">
                <span class="checkmark">✓</span>
            </div>
            <h3>Review Submitted Successfully</h3>
            <p>Thank you for your decision. The system will continue processing.</p>
        </div>
    </body>
    </html>
    '''


def get_free_port():
    """Find and return a free port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    addr, port = s.getsockname()
    s.close()
    return port


def create_review_app(state, result, page_status):
    """Create and configure the FastAPI app for human review."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def review_get():
        page_status['active'] = True
        page_status['last_heartbeat'] = time.time()
        image_path = state['image_path']
        image_url = f"/image?path={image_path.replace('\\', '/')}"
        return render_review_html(state['predicted_class'], state['reason'], image_url)

    @app.post("/", response_class=HTMLResponse)
    async def review_post(decision: str = Form(...)):
        result['decision'] = decision
        page_status['active'] = False
        return render_thank_you_html()
    
    @app.post("/heartbeat")
    async def heartbeat():
        page_status['last_heartbeat'] = time.time()
        return {"status": "ok"}
    
    @app.post("/page-closing")
    async def page_closing():
        page_status['page_closed'] = True
        return {"status": "ok"}

    @app.get("/image")
    async def get_image(path: str):
        if not path or not os.path.exists(path):
            return HTMLResponse(content="Image not found", status_code=404)
        return FileResponse(path)

    return app


def start_review_ui(state):
    """
    Start the human review UI and wait for user decision.
    Reopens the UI if closed without making a decision.
    
    Args:
        state: The state dictionary containing image_path, predicted_class, and reason
        
    Returns:
        str: The decision ('y' or 'n')
    """
    result = {'decision': None}
    
    while result['decision'] is None:
        page_status = {
            'active': False,
            'last_heartbeat': time.time(),
            'page_closed': False
        }
        
        # Create FastAPI app
        app = create_review_app(state, result, page_status)
        
        # Get a free port
        port = get_free_port()
        
        def run_app():
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
        
        # Start FastAPI app in a thread
        thread = threading.Thread(target=run_app)
        thread.daemon = True
        thread.start()
        
        # Wait a moment for server to start
        time.sleep(1)
        
        # Open browser to the correct port
        webbrowser.open(f'http://localhost:{port}/')
        
        # Wait for decision or detect if page was closed without decision
        while result['decision'] is None:
            time.sleep(0.5)
            
            # Check if page is still active based on heartbeat
            if page_status['active']:
                time_since_heartbeat = time.time() - page_status['last_heartbeat']
                
                # If no heartbeat for 5 seconds, page was likely closed
                if time_since_heartbeat > 5 or page_status['page_closed']:
                    print("\n⚠️  UI was closed without making a decision. Reopening...")
                    time.sleep(1)
                    break  # Exit inner loop to reopen UI
        
        # If decision was made, exit the outer loop
        if result['decision'] is not None:
            break
    
    return result['decision']

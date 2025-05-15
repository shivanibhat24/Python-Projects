<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Online IDE with Comment Remover</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .controls {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            gap: 10px;
        }
        
        .language-selector {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        select, button {
            padding: 8px 15px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        
        select {
            background-color: white;
        }
        
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        
        button:hover {
            background-color: #45a049;
        }
        
        .code-area {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .code-container {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .code-header {
            background-color: #f0f0f0;
            padding: 8px 15px;
            border-radius: 4px 4px 0 0;
            border: 1px solid #ddd;
            border-bottom: none;
            font-weight: bold;
        }
        
        textarea {
            flex: 1;
            min-height: 300px;
            padding: 15px;
            font-family: monospace;
            font-size: 14px;
            border: 1px solid #ddd;
            border-radius: 0 0 4px 4px;
            resize: vertical;
        }
        
        .button-container {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        
        .info-box {
            background-color: #e6f7ff;
            border-left: 4px solid #1890ff;
            padding: 10px 15px;
            margin-top: 20px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Online IDE with Comment & Print Remover</h1>
        
        <div class="controls">
            <div class="language-selector">
                <label for="language">Language:</label>
                <select id="language">
                    <option value="python">Python</option>
                    <option value="javascript">JavaScript</option>
                    <option value="java">Java</option>
                    <option value="cpp">C++</option>
                    <option value="csharp">C#</option>
                    <option value="php">PHP</option>
                    <option value="ruby">Ruby</option>
                </select>
            </div>
            <div>
                <button id="clearBtn">Clear</button>
            </div>
        </div>
        
        <div class="code-area">
            <div class="code-container">
                <div class="code-header">Original Code</div>
                <textarea id="codeInput" placeholder="Enter your code here..."></textarea>
            </div>
            
            <div class="code-container">
                <div class="code-header">Processed Code</div>
                <textarea id="codeOutput" placeholder="Result will appear here..." readonly></textarea>
            </div>
        </div>
        
        <div class="button-container">
            <button id="removeBtn">Remove Comments & Print Statements</button>
        </div>
        
        <div class="info-box">
            <p><strong>Note:</strong> This tool removes both comments and print statements from your code. 
               For multi-language support, it handles different comment styles and print functions based on the selected language.</p>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const languageSelect = document.getElementById('language');
            const codeInput = document.getElementById('codeInput');
            const codeOutput = document.getElementById('codeOutput');
            const removeBtn = document.getElementById('removeBtn');
            const clearBtn = document.getElementById('clearBtn');
            
            // Configure language-specific patterns
            const languagePatterns = {
                python: {
                    lineComment: '#',
                    blockCommentStart: '"""',
                    blockCommentEnd: '"""',
                    blockCommentAlt: {
                        start: "'''",
                        end: "'''"
                    },
                    printStatements: ['print\\s*\\(.*?\\)']
                },
                javascript: {
                    lineComment: '//',
                    blockCommentStart: '/*',
                    blockCommentEnd: '*/',
                    printStatements: ['console\\.log\\s*\\(.*?\\)', 'console\\.info\\s*\\(.*?\\)', 'console\\.debug\\s*\\(.*?\\)', 'console\\.warn\\s*\\(.*?\\)', 'console\\.error\\s*\\(.*?\\)']
                },
                java: {
                    lineComment: '//',
                    blockCommentStart: '/*',
                    blockCommentEnd: '*/',
                    printStatements: ['System\\.out\\.println\\s*\\(.*?\\)', 'System\\.out\\.print\\s*\\(.*?\\)', 'System\\.err\\.println\\s*\\(.*?\\)', 'System\\.err\\.print\\s*\\(.*?\\)']
                },
                cpp: {
                    lineComment: '//',
                    blockCommentStart: '/*',
                    blockCommentEnd: '*/',
                    printStatements: ['std::cout\\s*<<.*?;', 'std::cerr\\s*<<.*?;', 'printf\\s*\\(.*?\\);']
                },
                csharp: {
                    lineComment: '//',
                    blockCommentStart: '/*',
                    blockCommentEnd: '*/',
                    printStatements: ['Console\\.WriteLine\\s*\\(.*?\\)', 'Console\\.Write\\s*\\(.*?\\)', 'Debug\\.Log\\s*\\(.*?\\)']
                },
                php: {
                    lineComment: '//',
                    blockCommentStart: '/*',
                    blockCommentEnd: '*/',
                    blockCommentAlt: {
                        start: '#',
                        end: '\n'
                    },
                    printStatements: ['echo\\s+.*?;', 'print\\s+.*?;', 'var_dump\\s*\\(.*?\\);', 'print_r\\s*\\(.*?\\);']
                },
                ruby: {
                    lineComment: '#',
                    blockCommentStart: '=begin',
                    blockCommentEnd: '=end',
                    printStatements: ['puts\\s+.*?($|;)', 'print\\s+.*?($|;)', 'p\\s+.*?($|;)']
                }
            };
            
            // Function to remove comments and print statements
            function processCode(code, language) {
                const patterns = languagePatterns[language];
                let processedCode = code;
                
                if (!patterns) return code;
                
                // Remove block comments
                if (patterns.blockCommentStart && patterns.blockCommentEnd) {
                    const blockCommentRegex = new RegExp(escapeRegExp(patterns.blockCommentStart) + '[\\s\\S]*?' + escapeRegExp(patterns.blockCommentEnd), 'g');
                    processedCode = processedCode.replace(blockCommentRegex, '');
                    
                    // Handle alternative block comments if defined
                    if (patterns.blockCommentAlt) {
                        const altBlockCommentRegex = new RegExp(escapeRegExp(patterns.blockCommentAlt.start) + '[\\s\\S]*?' + escapeRegExp(patterns.blockCommentAlt.end), 'g');
                        processedCode = processedCode.replace(altBlockCommentRegex, '');
                    }
                }
                
                // Remove single-line comments
                if (patterns.lineComment) {
                    const lineCommentRegex = new RegExp(escapeRegExp(patterns.lineComment) + '.*?$', 'gm');
                    processedCode = processedCode.replace(lineCommentRegex, '');
                }
                
                // Remove print statements
                if (patterns.printStatements && patterns.printStatements.length) {
                    patterns.printStatements.forEach(pattern => {
                        const printRegex = new RegExp(pattern, 'g');
                        processedCode = processedCode.replace(printRegex, '');
                    });
                }
                
                // Remove extra blank lines and normalize spacing
                processedCode = processedCode.replace(/\n\s*\n+/g, '\n\n');
                
                return processedCode;
            }
            
            // Helper function to escape special characters in regex
            function escapeRegExp(string) {
                return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            }
            
            // Event listeners
            removeBtn.addEventListener('click', () => {
                const code = codeInput.value;
                const language = languageSelect.value;
                const processedCode = processCode(code, language);
                codeOutput.value = processedCode;
            });
            
            clearBtn.addEventListener('click', () => {
                codeInput.value = '';
                codeOutput.value = '';
            });
            
            // Add some sample code for Python by default
            codeInput.value = `# This is a sample Python script
def greet(name):
    """
    This function greets the person passed in as a parameter
    """
    print(f"Hello, {name}!")
    return f"Hello, {name}!"  # This return will be kept

# This is a comment that will be removed
print("Testing the comment remover")

class Calculator:
    # Calculator class for basic operations
    def __init__(self):
        print("Calculator initialized")
        
    def add(self, a, b):
        print(f"Adding {a} and {b}")
        return a + b

# More comments to be removed
result = greet("World")
print(f"Result: {result}")`;
        });
    </script>
</body>
</html>

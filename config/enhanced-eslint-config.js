// Enhanced ESLint Configuration with Maximum Auto-Fix Capabilities
// Optimized for TypeScript + React with comprehensive auto-fixing

module.exports = {
  root: true,
  parser: '@typescript-eslint/parser',
  plugins: [
    '@typescript-eslint',
    'react',
    'react-hooks'
  ],
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:@typescript-eslint/recommended',
    // NOTE: 'prettier' should be last to override formatting rules
  ],
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true
    }
  },
  settings: {
    react: {
      version: 'detect'
    }
  },
  env: {
    browser: true,
    es6: true,
    node: true,
    jest: true
  },
  rules: {
    // ============================================
    // AUTO-FIXABLE FORMATTING RULES
    // ============================================
    
    // Indentation and spacing
    'indent': ['error', 2, { 
      'SwitchCase': 1,
      'VariableDeclarator': 1,
      'outerIIFEBody': 1,
      'FunctionDeclaration': { 'parameters': 1, 'body': 1 },
      'FunctionExpression': { 'parameters': 1, 'body': 1 },
      'CallExpression': { 'arguments': 1 },
      'ArrayExpression': 1,
      'ObjectExpression': 1,
      'ImportDeclaration': 1,
      'flatTernaryExpressions': false,
      'ignoreComments': false
    }],
    
    // Quote and semicolon consistency
    'quotes': ['error', 'single', { 
      'avoidEscape': true, 
      'allowTemplateLiterals': false 
    }],
    'semi': ['error', 'always'],
    'semi-spacing': ['error', { 'before': false, 'after': true }],
    
    // Comma and bracket spacing
    'comma-dangle': ['error', 'always-multiline'],
    'comma-spacing': ['error', { 'before': false, 'after': true }],
    'object-curly-spacing': ['error', 'always'],
    'array-bracket-spacing': ['error', 'never'],
    'computed-property-spacing': ['error', 'never'],
    'key-spacing': ['error', { 'beforeColon': false, 'afterColon': true }],
    
    // Line breaks and whitespace
    'eol-last': ['error', 'always'],
    'no-trailing-spaces': 'error',
    'no-multiple-empty-lines': ['error', { 'max': 2, 'maxEOF': 1 }],
    'padded-blocks': ['error', 'never'],
    'space-before-blocks': 'error',
    'space-before-function-paren': ['error', {
      'anonymous': 'always',
      'named': 'never',
      'asyncArrow': 'always'
    }],
    'space-in-parens': ['error', 'never'],
    'space-infix-ops': 'error',
    'space-unary-ops': ['error', { 'words': true, 'nonwords': false }],
    
    // ============================================
    // AUTO-FIXABLE CODE QUALITY RULES
    // ============================================
    
    // Variable declarations and usage
    'no-unused-vars': 'off', // Handled by TypeScript version
    'prefer-const': ['error', { 'destructuring': 'any' }],
    'no-var': 'error',
    'one-var': ['error', 'never'],
    'one-var-declaration-per-line': ['error', 'always'],
    
    // Object and array improvements
    'object-shorthand': ['error', 'always'],
    'prefer-destructuring': ['error', {
      'VariableDeclarator': {
        'array': false,
        'object': true
      },
      'AssignmentExpression': {
        'array': true,
        'object': false
      }
    }],
    
    // Function improvements
    'prefer-arrow-callback': ['error', { 'allowNamedFunctions': false }],
    'arrow-spacing': ['error', { 'before': true, 'after': true }],
    'arrow-parens': ['error', 'avoid'],
    
    // String and template improvements  
    'prefer-template': 'error',
    'template-curly-spacing': ['error', 'never'],
    
    // Import/export formatting
    'sort-imports': ['error', {
      'ignoreCase': false,
      'ignoreDeclarationSort': false,
      'ignoreMemberSort': false,
      'memberSyntaxSortOrder': ['none', 'all', 'multiple', 'single'],
      'allowSeparatedGroups': false
    }],
    
    // Logical improvements
    'no-else-return': ['error', { 'allowElseIf': false }],
    'no-lonely-if': 'error',
    'no-unneeded-ternary': ['error', { 'defaultAssignment': false }],
    'no-nested-ternary': 'error',
    
    // ============================================
    // TYPESCRIPT SPECIFIC AUTO-FIXABLE RULES  
    // ============================================
    
    '@typescript-eslint/no-unused-vars': ['error', { 
      'argsIgnorePattern': '^_',
      'varsIgnorePattern': '^_',
      'caughtErrorsIgnorePattern': '^_'
    }],
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/prefer-optional-chain': 'error',
    '@typescript-eslint/prefer-nullish-coalescing': 'error',
    '@typescript-eslint/no-non-null-assertion': 'warn',
    
    // Type imports (auto-fixable in newer versions)
    '@typescript-eslint/consistent-type-imports': ['error', {
      'prefer': 'type-imports',
      'disallowTypeAnnotations': false
    }],
    
    // ============================================
    // REACT SPECIFIC AUTO-FIXABLE RULES
    // ============================================
    
    // Modern React patterns
    'react/react-in-jsx-scope': 'off', // Not needed in React 17+
    'react/prop-types': 'off', // Using TypeScript for prop validation
    
    // JSX formatting (auto-fixable)
    'react/jsx-uses-react': 'error',
    'react/jsx-uses-vars': 'error',
    'react/jsx-no-undef': 'error',
    'react/jsx-fragments': ['error', 'syntax'],
    'react/jsx-boolean-value': ['error', 'never'],
    'react/jsx-curly-brace-presence': ['error', {
      'props': 'never',
      'children': 'never'
    }],
    'react/jsx-equals-spacing': ['error', 'never'],
    'react/jsx-first-prop-new-line': ['error', 'multiline-multiprop'],
    'react/jsx-indent': ['error', 2],
    'react/jsx-indent-props': ['error', 2],
    'react/jsx-max-props-per-line': ['error', { 'maximum': 1, 'when': 'multiline' }],
    'react/jsx-no-useless-fragment': 'error',
    'react/jsx-props-no-multi-spaces': 'error',
    'react/jsx-sort-props': ['error', {
      'callbacksLast': false,
      'shorthandFirst': false,
      'shorthandLast': false,
      'ignoreCase': true,
      'noSortAlphabetically': false,
      'reservedFirst': true
    }],
    'react/jsx-tag-spacing': ['error', {
      'closingSlash': 'never',
      'beforeSelfClosing': 'always',
      'afterOpening': 'never',
      'beforeClosing': 'never'
    }],
    'react/jsx-wrap-multilines': ['error', {
      'declaration': 'parens-new-line',
      'assignment': 'parens-new-line',
      'return': 'parens-new-line',
      'arrow': 'parens-new-line',
      'condition': 'parens-new-line',
      'logical': 'parens-new-line',
      'prop': 'parens-new-line'
    }],
    
    // React Hooks rules (some auto-fixable)
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn'
  },
  overrides: [
    {
      // Test files have relaxed rules
      files: ['**/*.test.{ts,tsx}', '**/*.spec.{ts,tsx}'],
      env: {
        jest: true
      },
      rules: {
        'no-unused-vars': 'off',
        '@typescript-eslint/no-unused-vars': 'off',
        'prefer-const': 'off' // Tests often reassign variables
      }
    },
    {
      // Configuration files
      files: ['*.config.{js,ts}', '*.rc.{js,ts}'],
      rules: {
        'no-unused-vars': 'off',
        '@typescript-eslint/no-unused-vars': 'off'
      }
    }
  ]
};
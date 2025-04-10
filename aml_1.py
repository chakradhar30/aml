exp 1: 

import java.util.*;

public class StringEnding {
    public static void main(String args[]) {

        Scanner sc = new Scanner(System.in);
        System.out.println("1. Enter the string");
        System.out.println("2. Exit");
        System.out.println("Enter a choice");
        int n = sc.nextInt();
        
        while (n != 2) {
            System.out.println("Enter the string :");
            String s = sc.next();

            if (s.endsWith("abc")) {
                System.out.println(s + " is Accepted");
            } else {
                System.out.println(s + " is Not Accepted");
            }

            System.out.println("1. Enter the string\n2. Exit");
            System.out.println("Enter a choice");
            n = sc.nextInt();
        }
        sc.close();
    }
}

exp 2 :

import java.util.*;

public class LexicalAnalyzer {
    public static void main(String[] args) {
        ArrayList<String> keywords = new ArrayList<>(Arrays.asList(
            "if", "else", "while", "for", "int", "float", "double", "char", "String", "boolean"));

        ArrayList<String> operators = new ArrayList<>(Arrays.asList(
            "+", "-", "*", "/", "=", ">", "<", "!", "&", "|", "++", "--"));

        ArrayList<String> delimiters = new ArrayList<>(Arrays.asList(
            "(", ")", "{", "}", "[", "]", ",", ";"));

        Scanner sc = new Scanner(System.in);
        System.out.println("Enter program with single spaces");
        String input = sc.nextLine();
        String[] arr = input.split(" ");
        int len = arr.length;
        String[] ans = new String[len];

        for (int i = 0; i < len; i++) {
            if (keywords.contains(arr[i])) {
                ans[i] = "keyword";
            } else if (operators.contains(arr[i])) {
                ans[i] = "operator";
            } else if (delimiters.contains(arr[i])) {
                ans[i] = "delimiter";
            } else if (isIdentifier(arr[i])) {
                ans[i] = "identifier";
            } else if (isLiteral(arr[i])) {
                ans[i] = "literal";
            } else {
                ans[i] = "unknown";
            }
        }

        for (int i = 0; i < len; i++) {
            System.out.println(arr[i] + ": " + ans[i]);
        }

        sc.close();
    }

    private static boolean isIdentifier(String str) {
        if (Character.isDigit(str.charAt(0))) {
            return false;
        }
        for (char c : str.toCharArray()) {
            if (!Character.isLetterOrDigit(c) && c != '_') {
                return false;
            }
        }
        return true;
    }

    private static boolean isLiteral(String str) {
        try {
            Integer.parseInt(str);
            return true;
        } catch (NumberFormatException e1) {
            try {
                Double.parseDouble(str);
                return true;
            } catch (NumberFormatException e2) {
                return false;
            }
        }
    }
}


exp 3 :

import java.util.Scanner;

public class LeftRecursionElimination {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter Number of Productions: ");
        int num = scanner.nextInt();
        scanner.nextLine(); // Consume newline

        System.out.println("Enter the grammar as A -> Aa / b:");
        for (int i = 0; i < num; i++) {
            String production = scanner.nextLine().trim();
            eliminateLeftRecursion(production);
        }

        scanner.close();
    }

    public static void eliminateLeftRecursion(String production) {
        String[] parts = production.split("->");
        char nonTerminal = parts[0].charAt(0);
        String[] choices = parts[1].split("/");

        System.out.println("GRAMMAR: " + production);

        // Checking for left recursion
        if (choices[0].startsWith("" + nonTerminal)) {
            String beta = choices[0].substring(1); // Extracting beta from the first choice
            System.out.println(nonTerminal + " is left recursive.");

            // Printing reduced grammar
            System.out.println(nonTerminal + " -> " + choices[1].trim() + nonTerminal + "'");
            System.out.println(nonTerminal + "' -> " + beta + nonTerminal + "' / epsilon");
        } else {
            System.out.println(nonTerminal + " is not left recursive.");
        }
    }
}



exp 4:

 import java.util.*;

public class LeftFactoring {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter the number of productions:");
        int n = sc.nextInt();
        sc.nextLine();
        
        String[] productions = new String[n];
        System.out.println("Enter the productions:");
        for (int i = 0; i < n; i++) {
            productions[i] = sc.nextLine();
        }
        
        eliminateLeftFactoring(productions);
        sc.close();
    }

    private static void eliminateLeftFactoring(String[] productions) {
        boolean leftFactored = false;
        
        for (String production : productions) {
            String[] parts = production.split("->");
            String lhs = parts[0].trim();
            String[] rhs = parts[1].split("\\|");
            String prefix = findCommonPrefix(rhs);

            if (!prefix.isEmpty()) {
                leftFactored = true;
                System.out.println(lhs + "->" + prefix + lhs + "'");

                List<String> newRhs = new ArrayList<>();
                for (String r : rhs) {
                    if (r.startsWith(prefix)) {
                        String suffix = r.substring(prefix.length()).trim();
                        if (suffix.isEmpty()) {
                            suffix = "";
                        }
                        newRhs.add(suffix);
                    } else {
                        newRhs.add(r);
                    }
                }
                System.out.println(lhs + "'->" + String.join("|", newRhs));
            }
        }
        
        if (!leftFactored) {
            System.out.println("Given productions do not have left factoring");
        }
    }

    private static String findCommonPrefix(String[] rhs) {
        String prefix = rhs[0];
        
        for (int i = 1; i < rhs.length; i++) {
            while (rhs[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);
                if (prefix.isEmpty()) {
                    return "";
                }
            }
        }
        return prefix;
    }
}
 
    

exp 5 :

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

public class first {
    static String first[], follow[], grammar[][];
    static List<String> nonTerminals = new ArrayList<>();

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter the no of productions");
        int n = sc.nextInt();
        grammar = new String[n][2];
        System.out.println("Enter the productions that are free from Left Recursion");
        sc.nextLine();
        
        for (int i = 0; i < n; i++) {
            String s = sc.nextLine();
            String p[] = s.split("->");
            nonTerminals.add(p[0].trim());
            grammar[i][0] = p[0].trim();
            grammar[i][1] = p[1];
        }

        first = new String[n];
        follow = new String[n];
        
        for (int i = 0; i < n; i++)
            first[i] = calculateFirst(i);
        
        System.out.println("First :-");
        for (int i = 0; i < n; i++)
            System.out.println(nonTerminals.get(i) + " : " + print(first[i]));

        for (int i = 0; i < n; i++)
            follow[i] = calculateFollow(i);
        
        System.out.println("Follow :-");
        for (int i = 0; i < n; i++)
            System.out.println(nonTerminals.get(i) + " : " + print(follow[i]));
    }

    static String print(String s) {
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        sb.append(s.charAt(0));
        for (char c : s.toCharArray())
            if (sb.indexOf(c + "") == -1)
                sb.append("," + c);
        sb.append('}');
        return sb.toString();
    }

    static String calculateFirst(int i) {
        String s[] = grammar[i][1].split("\\|"), temp = "";
        for (String p : s) {
            if (!nonTerminals.contains(p.charAt(0) + ""))
                temp += p.charAt(0);
            else
                temp += calculateFirst(nonTerminals.indexOf(p.charAt(0) + ""));
        }
        return temp;
    }

    static String calculateFollow(int i) {
        Set<Character> temp = new HashSet<>();
        if (i == 0)
            temp.add('$');

        for (int idx = 0; idx < grammar.length; idx++) {
            if (grammar[idx][0] == nonTerminals.get(i))
                continue;

            String s[] = grammar[idx][1].split("\\|");
            for (String p : s) {
                String nt = nonTerminals.get(i);
                if (p.contains(nt)) {
                    if (p.indexOf(nt) == p.length() - 1) {
                        String t = follow[nonTerminals.indexOf(grammar[idx][0])];
                        for (char c : t.toCharArray())
                            temp.add(c);
                    } else {
                        int x = p.indexOf(nt);
                        if (!nonTerminals.contains(p.charAt(x + 1) + ""))
                            temp.add(p.charAt(x + 1));
                        else {
                            if (first[nonTerminals.indexOf(p.charAt(x + 1) + "")].contains("e")) {
                                String t = first[nonTerminals.indexOf(p.charAt(x + 1) + "")];
                                for (char c : t.toCharArray())
                                    temp.add(c);
                                temp.remove('e');
                                t = follow[nonTerminals.indexOf(grammar[idx][0])];
                                for (char c : t.toCharArray())
                                    temp.add(c);
                            } else {
                                String t = first[nonTerminals.indexOf(p.charAt(x + 1) + "")];
                                for (char c : t.toCharArray())
                                    temp.add(c);
                            }
                        }
                    }
                }
            }
        }

        String ans = "";
        for (char x : temp)
            ans += x;
        return ans;
    }
}


exp 6 :


import java.util.*;

public class Parse {
    static String s = "", st = "";
    
    @SuppressWarnings("resource")
    public static void main(String[] args) {
        String[][] tab = {
            {"ta", "@", "@", "ta", "@", "@"},
            {"@", "+ta", "@", "@", "!", "!"},
            {"fb", "@", "@", "fb", "@", "@"},
            {"@", "!", "*fb", "@", "!", "!"},
            {"i", "@", "@", "(e)", "@", "@"}
        };
        
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the string:\n");
        s = scanner.nextLine();
        s += "$";
        st = "$e";
        
        int st_i = 1, s_i = 0;
        StringBuilder temp = new StringBuilder();
        
        System.out.println("\nStack\t\tInput");
        
        while (!st.endsWith("$") || !s.endsWith("$")) {
            int s1, s2;
            
            switch (st.charAt(st_i)) {
                case 'e': s1 = 0; break;
                case 'a': s1 = 1; break;
                case 't': s1 = 2; break;
                case 'b': s1 = 3; break;
                case 'f': s1 = 4; break;
                default: s1 = -1;
            }
            
            switch (s.charAt(s_i)) {
                case 'i': s2 = 0; break;
                case '+': s2 = 1; break;
                case '*': s2 = 2; break;
                case '(': s2 = 3; break;
                case ')': s2 = 4; break;
                case '$': s2 = 5; break;
                default: s2 = -1;
            }
            
            if (s1 == -1 || s2 == -1 || tab[s1][s2].equals("@")) {
                System.out.println("Failure");
                return;
            }
            
            if (tab[s1][s2].startsWith("!")) {
                st = st.substring(0, st_i);
                st_i--;
            } else {
                temp.setLength(0);
                for (int k = tab[s1][s2].length() - 1; k >= 0; k--) {
                    temp.append(tab[s1][s2].charAt(k));
                }
                st = st.substring(0, st_i) + temp.toString();
                st_i = st.length() - 1;
            }
            
            System.out.print(st + "\t\t");
            for (int n = s_i; n < s.length(); n++) {
                System.out.print(s.charAt(n));
            }
            System.out.println();
            
            if (st.charAt(st_i) == s.charAt(s_i) && s.charAt(s_i) != '$') {
                st = st.substring(0, st_i);
                s_i++;
                st_i--;
            }
        }
        
        System.out.println("Success");
        scanner.close();
    }
}


exp 7 :

import java.util.Scanner;

class ProductionRule {
    String left;
    String right;

    ProductionRule(String left, String right) {
        this.left = left;
        this.right = right;
    }
}

public class Exp7 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String input, stack = ""; 
        int ruleCount;

        System.out.println("Enter the number of production rules: ");
        ruleCount = scanner.nextInt();
        scanner.nextLine();

        ProductionRule[] rules = new ProductionRule[ruleCount];
        System.out.println("Enter the production rules (in the form 'left->right'): ");
        for (int i = 0; i < ruleCount; i++) {
            String[] temp = scanner.nextLine().split("->");
            rules[i] = new ProductionRule(temp[0], temp[1]);
        }

        System.out.println("Enter the input string: ");
        input = scanner.nextLine();
        int i = 0;

        System.out.println("Stack\tInputBuffer\tAction");
        while (true) {
            if (i < input.length()) {
                char ch = input.charAt(i);
                i++;
                stack += ch;
                System.out.print(stack + "\t");
                System.out.print(input.substring(i) + "\t\tShift " + ch + "\n");
            }
            
            for (int j = 0; j < ruleCount; j++) {
                int substringIndex = stack.indexOf(rules[j].right);
                if (substringIndex != -1) {
                    stack = stack.substring(0, substringIndex) + rules[j].left;
                    System.out.print(stack + "\t");
                    System.out.print(input.substring(i) + "\t\tReduce " + rules[j].left + "->" + rules[j].right + "\n");
                    j = -1;
                }
            }
            
            if (stack.equals(rules[0].left) && i == input.length()) {
                System.out.println("\nAccepted");
                break;
            }
            
            if (i == input.length()) {
                System.out.println("\nNot Accepted");
                break;
            }
        }
        scanner.close();
    }
}



exp 8 :


import java.util.Scanner;

public class Exp8 {
    public static void main(String[] args) {
        char[] stack = new char[20];
        char[] ip = new char[20];
        char[][][] opt = new char[10][10][1];
        char[] ter = new char[10];
        int i, j, k, n, top = 0, col = 0, row = 0;
        Scanner scanner = new Scanner(System.in);

        for (i = 0; i < 10; i++) {
            stack[i] = 0;
            ip[i] = 0;
            for (j = 0; j < 10; j++) {
                opt[i][j][0] = 0;
            }
        }

        System.out.print("Enter the no. of terminals: ");
        n = scanner.nextInt();
        System.out.print("\nEnter the terminals: ");
        ter = scanner.next().toCharArray();

        System.out.println("\nEnter the table values:");
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                System.out.printf("\nEnter the value for %c %c: ", ter[i], ter[j]);
                opt[i][j] = scanner.next().toCharArray();
            }
        }

        System.out.println("\nOPERATOR PRECEDENCE TABLE:");
        for (i = 0; i < n; i++) {
            System.out.print("\t" + ter[i]);
        }
        System.out.println();

        for (i = 0; i < n; i++) {
            System.out.println();
            System.out.print(ter[i]);
            for (j = 0; j < n; j++) {
                System.out.print("\t" + opt[i][j][0]);
            }
        }

        stack[top] = '$';
        System.out.print("\nEnter the input string: ");
        String input = scanner.next();
        ip = input.toCharArray();
        i = 0;

        System.out.println("\nSTACK\t\t\tINPUT STRING\t\t\tACTION");
        System.out.print("\n" + String.valueOf(stack) + "\t" + input + "\t\t");

        while (i <= input.length()) {
            for (k = 0; k < n; k++) {
                if (stack[top] == ter[k]) col = k;
                if (ip[i] == ter[k]) row = k;
            }

            if ((stack[top] == '$') && (ip[i] == '$')) {
                System.out.println("String is accepted");
                break;
            } else if ((opt[col][row][0] == '<') || (opt[col][row][0] == '=')) {
                stack[++top] = opt[col][row][0];
                 stack[++top] = ip[i];
                System.out.println("Shift " + ip[i]);
                i++;
            } else {
                if (opt[col][row][0] == '>') {
                    while (stack[top] != '<') {
                        --top;
                    }
                    top = top - 1;
                    System.out.println("Reduce");
                } else {
                    System.out.println("\nString is not accepted");
                    break;
                }
            }

            System.out.println();
            for (k = 0; k <= top; k++) {
                System.out.print(stack[k]);
            }
            System.out.print("\t\t\t");
            for (k = i; k < input.length(); k++) {
                System.out.print(ip[k]);
            }
            System.out.print("\t\t\t");
        }
        scanner.close();
    }
}


exp 9 :


import java.util.Scanner;

public class Exp9 {
    static int[] stack;
    static int top, n;

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        top = -1;

        System.out.println("Enter the size of stack[MAX=100]: ");
        n = scanner.nextInt();

        if (n <= 0) {
            System.out.println("Invalid stack size.");
            return;
        }

        stack = new int[n];

        System.out.println("\n\tStack Operations:");
        System.out.println("\t--------------------------");
        System.out.println("\t1. Push");
        System.out.println("\t2. Pop");
        System.out.println("\t3. Display");
        System.out.println("\t4. EXIT");

        int choice;
        do {
            System.out.println("\nEnter your choice: ");
            choice = scanner.nextInt();

            switch (choice) {
                case 1:
                    push(scanner);
                    break;
                case 2:
                    pop();
                    break;
                case 3:
                    display();
                    break;
                case 4:
                    System.out.println("\nEXIT");
                    break;
                default:
                    System.out.println("Please enter a valid choice.");
            }
        } while (choice != 4);

        scanner.close();
    }

    static void push(Scanner scanner) {
        if (top >= n - 1) {
            System.out.println("\nStack overflow");
        } else {
            System.out.println("Enter a value to be pushed: ");
            int x = scanner.nextInt();
            
            stack[++top] = x;
        }
    }

    static void pop() {
        if (top == -1) {
            System.out.println("\nStack underflow");
        } else {
            System.out.println("\nThe popped element is " + stack[top--]);
            
        }
    }

    static void display() {
        if (top >= 0) {
            System.out.println("\nThe elements in the stack are:");
            for (int i = top; i >= 0; i--) {
                System.out.println(stack[i]);
            }
            System.out.println("\nSelect next choice");
        } else {
            System.out.println("\nThe stack is empty.");
        }
    }
}



exp 10 :

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Stack;

class Instruction {
    String op;
    String arg1;
    String arg2;
    String result;

    Instruction(String op, String arg1, String arg2, String result) {
        this.op = op;
        this.arg1 = arg1;
        this.arg2 = arg2;
        this.result = result;
    }

    @Override
    public String toString() {
        return result + " = " + arg1 + " " + op + " " + arg2;
    }
}

class IntermediateCodeGenerator {
    private List<Instruction> instructions;
    private Stack<String> operands;
    private int tempCount;

    IntermediateCodeGenerator() {
        instructions = new ArrayList<>();
        operands = new Stack<>();
        tempCount = 0;
    }

    public List<Instruction> generate(String expression) {
        Stack<Character> operators = new Stack<>();
        StringBuilder operand = new StringBuilder();

        for (int i = 0; i < expression.length(); i++) {
            char token = expression.charAt(i);

            if (Character.isWhitespace(token)) {
                continue;
            }

            if (Character.isLetterOrDigit(token)) {
                operand.append(token);
                if (i == expression.length() - 1 || !Character.isLetterOrDigit(expression.charAt(i + 1))) {
                    operands.push(operand.toString());
                    operand.setLength(0);
                }
            } else if (token == '(') {
                operators.push(token);
            } else if (token == ')') {
                while (!operators.isEmpty() && operators.peek() != '(') {
                    processOperator(operators.pop());
                }
                operators.pop(); // remove '('
            } else if (isOperator(token)) {
                while (!operators.isEmpty() && precedence(token) <= precedence(operators.peek())) {
                    processOperator(operators.pop());
                }
                operators.push(token);
            }
        }

        while (!operators.isEmpty()) {
            processOperator(operators.pop());
        }

        return instructions;
    }

    private void processOperator(char operator) {
        String operand2 = operands.pop();
        String operand1 = operands.pop();
        String result = newTemp();
        instructions.add(new Instruction(String.valueOf(operator), operand1, operand2, result));
        operands.push(result);
    }

    private String newTemp() {
        return "t" + tempCount++;
    }

    private boolean isOperator(char token) {
        return token == '+' || token == '-' || token == '*' || token == '/';
    }

    private int precedence(char operator) {
        switch (operator) {
            case '+':
            case '-':
                return 1;
            case '*':
            case '/':
                return 2;
            default:
                return -1;
        }
    }
}

public class exp10 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter an arithmetic expression:");
        String expression = scanner.nextLine();

        IntermediateCodeGenerator icg = new IntermediateCodeGenerator();
        List<Instruction> code = icg.generate(expression);

        System.out.println("Intermediate Code:");
        for (Instruction instr : code) {
            System.out.println(instr);
        }

        scanner.close();
    }
}

package tull.annet;

import java.util.Scanner;


class Curser {

    public static void main(String[] args) {

        int maksTimer = 4;
        int currentTimer = 0;

        // user input
        Scanner scanner = new Scanner(System.in);
        System.out.print("\nVil du starte webm?\n> ");
        String inp = scanner.nextLine();

        // while loop som kjører saå lenge bruker svarer "ja" paa input
        while (inp.equalsIgnoreCase("ja")) {

            try {
                Thread.sleep(800);
                System.out.println();
                System.out.println("Current time: "+currentTimer);
                System.out.println("Maks time:    "+maksTimer);
                currentTimer++;
                maksTimer++;
            }
            catch (InterruptedException _) {}
        }
        System.out.println();
    }
}
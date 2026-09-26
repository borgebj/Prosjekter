package tull.kryptering;

import java.util.Scanner;

class Krypt {

    public static void main(String[] args) {

        System.out.println("----------------------------------------\n");

        // Scanner-objekt
        Scanner scanner = new Scanner(System.in);

        // input fra bruker av type string i variabel "passord"
        System.out.print("Hva er ditt passord: ");
        String passord = scanner.nextLine();

        StringBuilder kryptert = new StringBuilder();

        for (int i = 0; i < passord.length(); i++) {
            kryptert.append((char) (passord.charAt(i) - +3));
        }

        // skriver ut brukerinput for test
        System.out.println("Ditt passord er: "+kryptert);


        System.out.println("\n----------------------------------------");

    }
}
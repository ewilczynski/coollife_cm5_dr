pipeline {
    agent any

    stages {
        stage('Init') {
            steps {
                checkout scm
            }
        }

        stage('Build & Test') {
            steps {
                script {
                    env.REPO_NAME = getRepoName()
                    try {
                        sh "docker-compose -f docker-compose.tests.yml -p visualization_tool up --build --exit-code-from $REPO_NAME"
                    } finally {
                        sh 'docker-compose -f docker-compose.tests.yml down -v --rmi all --remove-orphans'
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    env.COMMIT_ID = sh(returnStdout: true, script: 'git rev-parse HEAD').trim()
                    env.REPO_NAME = getRepoName()
                    env.BRANCH_NAME = scm.branches[0].name

                    echo "Deploying commit $COMMIT_ID of repository $REPO_NAME on branch $BRANCH_NAME"

                    def remote = [:]
                    withCredentials([string(credentialsId: '2be3945e-1845-4fd0-af62-85fa3cefd8da', variable: 'SERVER_REMOTE_NAME')]) {
                        remote.name = SERVER_REMOTE_NAME
                    }
                    withCredentials([string(credentialsId: 'c1b06132-e298-46f6-bbf0-b5ae53be7556', variable: 'SERVER_REMOTE_HOST')]) {
                        remote.host = SERVER_REMOTE_HOST
                    }
                    remote.allowAnyHosts = true
                    withCredentials([usernamePassword(credentialsId: 'b6b0a17f-f99c-4750-8e46-411427b3e5b8', passwordVariable: 'password', usernameVariable: 'username')]) {
                        remote.user = username
                        remote.password = password
                    }
                    sshCommand remote: remote, command: "cd /srv/citiwatts/ && sudo ./deploy_cm.sh ${env.REPO_NAME} ${env.COMMIT_ID}", returnStatus: true
                }
            }
        }
    }

    post {
        success {
            echo "SUCCESS"
        }

        failure {
            echo "FAILED"
            emailext(
                body: 'Check console output at $JOB_URL to view the results',
                to: 'support@easilabdev.ch',
                subject: 'Jenkins pipeline failed: $PROJECT_NAME - #$BUILD_NUMBER'
            )
        }
    }
}

String getRepoName() {
    return scm.getUserRemoteConfigs()[0].getUrl().tokenize('/').last().split("\\.")[0]
}
